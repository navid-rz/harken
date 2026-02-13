from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import os
import time
import argparse
from copy import deepcopy
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import csv

from train.utils import (
    _binary_counts, _derive_metrics, _multiclass_confusion_add,
    _multiclass_macro_prf1, compute_confusion_matrix,
)
from config import load_config
from model.model import DilatedTCN

from data_loader.utils import make_datasets, FeatureAugment
from analysis.plot_metrics import plot_metrics, plot_test_confusion_matrix
from train.evaluate import evaluate_model


# -----------------------
# Trainer
# -----------------------
class Trainer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        # Use task.type from config
        self.task_type = cfg["task"]["type"]  # "binary" or "multiclass"
        # Device
        device_cfg = cfg["train"].get("device", "auto")
        if device_cfg == "cuda":
            self.device = torch.device("cuda")
        elif device_cfg == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pin memory for faster GPU transfer (only beneficial with CUDA)
        self.pin_memory = self.device.type == "cuda"

        # Build datasets/loaders with augmentation flag
        aug_cfg = cfg.get("augmentation", {})
        use_aug = bool(aug_cfg.get("enable", False))
        augment = None
        if use_aug:
            # Try feature-specific hop length, fall back to mfcc for backward compatibility
            feature_cfg = cfg.get("data", {}).get("features", cfg.get("data", {}).get("mfcc", {}))
            hop_s = float(feature_cfg.get("hop_length_s", 0.01))
            augment = FeatureAugment(
                hop_length_s=hop_s,
                max_shift_ms=float(aug_cfg.get("max_shift_ms", 100.0)),
                noise_prob=float(aug_cfg.get("noise_prob", 0.15)),
                noise_std_factor=float(aug_cfg.get("noise_std_factor", 0.05)),
                seed=aug_cfg.get("seed", None),
            )
        
        self.train_loader, self.val_loader, self.test_loader = make_datasets(
            self.cfg, which="all", batch_size=self.cfg["train"]["batch_size"], train_transform=augment
        )
        self.train_dataset = self.train_loader.dataset
        self.val_dataset   = self.val_loader.dataset
        self.test_dataset  = self.test_loader.dataset

        # Get number of classes from dataset
        self.num_classes = self.train_dataset.num_classes
        print(f"[INFO] Detected {self.num_classes} classes")

        # Loaders already created by make_datasets
        self.batch_size = int(cfg["train"]["batch_size"])

        # Build model via shared constructor
        sample_x, _ = self.train_dataset[0]  # (C, T)
        if sample_x.dim() == 3:  # safety if any wrapper returns (B,C,T)
            sample_x = sample_x[0]
        C, T = int(sample_x.shape[0]), int(sample_x.shape[1])
        print(f"[Feature dims] channels(C)={C}  timesteps(T)={T}")
        self.model = DilatedTCN.from_config(cfg).to(self.device)
        self.input_shape = (C, T)  # save for checkpoint metadata

        # Optimizer
        lr = float(cfg["train"]["learning_rate"])
        weight_decay = float(cfg["train"]["weight_decay"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Weight constraints configuration
        self.constrain_weights = bool(cfg["train"].get("constrain_weights", False))
        self.weight_min = float(cfg["train"].get("weight_min", -1.0))
        self.weight_max = float(cfg["train"].get("weight_max", 1.0))
        print(f"[Weight Constraints] Enabled: {self.constrain_weights}, Range: [{self.weight_min}, {self.weight_max}]")
        
        # Activation constraints configuration
        self.constrain_act = bool(cfg["train"].get("constrain_act", False))
        self.act_min = float(cfg["train"].get("act_min", 0.0))
        self.act_max = float(cfg["train"].get("act_max", 1024.0))
        self.activation_hooks = []  # Store hook handles for cleanup
        print(f"[Activation Constraints] Enabled: {self.constrain_act}, Range: [{self.act_min}, {self.act_max}]")
        
        if self.constrain_act:
            self._apply_activation_constraint_hooks()

        # Loss
        if self.task_type == "binary":
            pos_weight = cfg["train"].get("pos_weight", None)
            if pos_weight is not None:
                pos_weight_tensor = torch.tensor([float(pos_weight)], device=self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
            self.threshold = float(cfg["train"].get("threshold", 0.5) or 0.5)
            self.auto_threshold = bool(cfg["train"].get("auto_threshold", False))
            self.threshold_search_steps = int(cfg["train"].get("threshold_search_steps", 101))
            self.class_names = ["neg", "pos"]
        else:
            class_weights = cfg["train"].get("class_weights", None)
            label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
            weight_tensor = None
            if class_weights is not None:
                if len(class_weights) != self.num_classes:
                    raise ValueError("class_weights length must equal number of classes.")
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
            self.class_names = self.train_dataset.class_names  # list[str] length = num_classes

        self.test_metrics = None
        self.test_confusion = None  # (only for multiclass)
        # Histories / outputs
        self.epochs = int(cfg["train"]["num_epochs"])
        self.plots_dir = os.path.join(cfg["output"]["plots_dir"], "training")
        self.metrics_figure = os.path.join(self.plots_dir, "metrics.png")
        self.weights_path = os.path.join(cfg["output"]["weights_dir"], "model_weights_fp.pt")
        self.ckpt_path    = os.path.join(cfg["output"]["weights_dir"], "model_ckpt_fp.pt")
        self.use_tqdm = bool(cfg["output"].get("tqdm", True))

        self.train_hist = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
        self.val_hist   = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
        # CSV logging path
        self.csv_path = os.path.join(self.plots_dir, "metrics.csv")
        self._init_csv_log()

        # Early stopping configuration
        early_stop_cfg = cfg["train"].get("early_stopping", {})
        self.early_stopping_enabled = bool(early_stop_cfg.get("enabled", False))
        self.early_stopping_patience = int(early_stop_cfg.get("patience", 10))
        self.early_stopping_metric = str(early_stop_cfg.get("metric", "val_loss"))
        self.early_stopping_min_delta = float(early_stop_cfg.get("min_delta", 0.001))
        self.early_stopping_restore_best = bool(early_stop_cfg.get("restore_best", True))
        
        # Early stopping state
        self.best_metric_value = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.best_model_state = None
        self.early_stopped = False

        # ---- Sanity checks: dataset sizes and feature shapes ----
        print(f"[Split sizes] train={len(self.train_dataset)}  val={len(self.val_dataset)}  test={len(self.test_dataset)}")

        # Peek a few samples from the training subset
        for i in range(min(3, len(self.train_dataset))):
            x_i, y_i = self.train_dataset[i]
            print(f"[Sample {i}] x.shape={tuple(x_i.shape)}  label={y_i}")

        # Print total number of trainable weights
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[TCN] Total trainable weights: {total_params}")

        # Check dataset balance/distribution
        if self.task_type == "binary":
            ys = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            pos = int(np.sum(ys)); neg = len(ys) - pos
            print(f"[Train balance] pos={pos}  neg={neg}  ratio={neg/max(1,pos):.2f}:1")
        
        if self.task_type == "multiclass":
            ys = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            counts = np.bincount(ys, minlength=self.num_classes)
            print(f"[Train per-class counts] {counts.tolist()}")
            print(f"[Classes] {self.class_names}")

        # Optional: assert reasonable ranges (adjust to your expectations)
        assert C in (16, 20, 28, 40, 64, 80, 128) or C > 0, f"Unexpected feature channels: {C}"
        assert T >= 40, f"Too few time steps ({T}). Check feature extraction hop/window/duration."

    def _loop(self, loader: DataLoader, train_mode: bool = True) -> Tuple[float, float, float, float, float]:
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        tp = fp = fn = 0
        if self.task_type == "multiclass":
            cm = [[0 for _ in range(self.num_classes)] for _ in range(self.num_classes)]
        else:
            cm = None

        iterator = tqdm(loader, desc="Training" if train_mode else "Validating", leave=False) if self.use_tqdm else loader

        with torch.set_grad_enabled(train_mode):
            for batch_x, batch_y in iterator:
                num_batches += 1
                batch_x = batch_x.to(self.device, non_blocking=self.pin_memory)
                
                # Scale input features to activation constraint range if enabled
                if self.constrain_act:
                    # Get current min/max of batch features
                    batch_min = batch_x.min()
                    batch_max = batch_x.max()
                    # Scale features to [act_min, act_max] range
                    if batch_max > batch_min:  # Avoid division by zero
                        batch_x = (batch_x - batch_min) / (batch_max - batch_min)  # Normalize to [0,1]
                        batch_x = batch_x * (self.act_max - self.act_min) + self.act_min  # Scale to [act_min, act_max]

                if self.task_type == "binary":
                    targets = batch_y.float().unsqueeze(1).to(self.device)
                else:
                    targets = batch_y.long().to(self.device)

                logits = self.model(batch_x)
                loss = self.criterion(logits, targets)

                if train_mode:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Apply weight constraints if enabled
                    if self.constrain_weights:
                        self._constrain_weights()

                total_loss += loss.item()

                if self.task_type == "binary":
                    probs = torch.sigmoid(logits)
                    preds = (probs > self.threshold).long()
                    correct += (preds == targets.long()).sum().item()
                    total += targets.numel()
                    _tp, _fp, _fn, _, _ = _binary_counts(preds, targets)
                    tp += _tp; fp += _fp; fn += _fn
                else:
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.numel()
                    cm = _multiclass_confusion_add(cm, preds, targets, self.num_classes)

        if self.task_type == "binary":
            return _derive_metrics(total_loss, num_batches, correct, total, tp, fp, fn)
        else:
            avg_loss, acc = _derive_metrics(total_loss, num_batches, correct, total)
            macro_p, macro_r, macro_f1 = _multiclass_macro_prf1(cm)
            return avg_loss, acc, macro_p, macro_r, macro_f1

    def _constrain_weights(self) -> None:
        """Constrain all weight parameters to specified range for hardware constraints."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() > 1:  # Only weight matrices, not biases
                    param.clamp_(self.weight_min, self.weight_max)
                    
    def _apply_activation_constraint_hooks(self) -> None:
        """Apply forward hooks to constrain activations between act_min and act_max."""
        def constraint_hook(module, input, output):
            # Constrain output to [act_min, act_max] range
            return torch.clamp(output, self.act_min, self.act_max)
        
        # Apply hooks to all ReLU layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                handle = module.register_forward_hook(constraint_hook)
                self.activation_hooks.append(handle)
                print(f"[Activation Constraints] Applied hook to {name}")
                
    def _remove_activation_constraint_hooks(self) -> None:
        """Remove all activation constraint hooks."""
        for handle in self.activation_hooks:
            handle.remove()
        self.activation_hooks.clear()
        print(f"[Activation Constraints] Removed all hooks")

    # ---------- Threshold search (binary only) ----------
    def find_best_threshold(self, steps: int = 101) -> Tuple[float, float]:
        assert self.task_type == "binary", "Threshold search is for binary mode."
        self.model.eval()
        probs_all, y_all = [], []
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.float().unsqueeze(1).to(self.device)
                logits = self.model(batch_x)
                probs_all.append(torch.sigmoid(logits).cpu())
                y_all.append(batch_y.cpu())
        probs_all = torch.cat(probs_all, dim=0).view(-1)
        y_all = torch.cat(y_all, dim=0).view(-1)

        best_thr, best_f1 = 0.5, -1.0
        for thr in torch.linspace(0.01, 0.99, steps=steps):
            preds = (probs_all > thr).int()
            tp = int(((preds == 1) & (y_all == 1)).sum())
            fp = int(((preds == 1) & (y_all == 0)).sum())
            fn = int(((preds == 0) & (y_all == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_thr = float(f1), float(thr)
        return best_thr, best_f1

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float, float, float]:
        return evaluate_model(
            self.model, loader, self.device, self.task_type,
            self.num_classes, self.criterion,
            threshold=getattr(self, "threshold", 0.5),
            pin_memory=self.pin_memory
        )
 
    def _check_early_stopping(self, va_loss: float, va_acc: float, va_f1: float, epoch: int) -> bool:
        """Check if early stopping criteria are met. Returns True if training should stop."""
        if not self.early_stopping_enabled:
            return False
            
        # Get current metric value based on configured metric
        if self.early_stopping_metric == "val_loss":
            current_value = va_loss
            is_better = lambda curr, best: curr < best - self.early_stopping_min_delta
        elif self.early_stopping_metric == "val_acc":
            current_value = va_acc
            is_better = lambda curr, best: curr > best + self.early_stopping_min_delta
        elif self.early_stopping_metric == "val_f1":
            current_value = va_f1
            is_better = lambda curr, best: curr > best + self.early_stopping_min_delta
        else:
            print(f"[WARNING] Unknown early stopping metric: {self.early_stopping_metric}. Disabling early stopping.")
            return False
            
        # Initialize best value on first epoch
        if self.best_metric_value is None:
            self.best_metric_value = current_value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            if self.early_stopping_restore_best:
                self.best_model_state = deepcopy(self.model.state_dict())
            return False
            
        # Check if current value is better than best
        if is_better(current_value, self.best_metric_value):
            self.best_metric_value = current_value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            if self.early_stopping_restore_best:
                self.best_model_state = deepcopy(self.model.state_dict())
            print(f"[Early Stopping] New best {self.early_stopping_metric}: {current_value:.4f} at epoch {epoch}")
        else:
            self.epochs_without_improvement += 1
            
        # Check if patience exceeded
        if self.epochs_without_improvement >= self.early_stopping_patience:
            print(f"[Early Stopping] No improvement for {self.early_stopping_patience} epochs. Best {self.early_stopping_metric}: {self.best_metric_value:.4f} at epoch {self.best_epoch}")
            if self.early_stopping_restore_best and self.best_model_state is not None:
                print(f"[Early Stopping] Restoring best model weights from epoch {self.best_epoch}")
                self.model.load_state_dict(self.best_model_state)
            return True
            
        return False

    def train(self) -> None:
        start = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}:")
            tr_loss, tr_acc, tr_p, tr_r, tr_f1 = self._loop(self.train_loader, train_mode=True)
            va_loss, va_acc, va_p, va_r, va_f1 = self._loop(self.val_loader,   train_mode=False)

            self.train_hist["loss"].append(tr_loss); self.val_hist["loss"].append(va_loss)
            self.train_hist["acc"].append(tr_acc);   self.val_hist["acc"].append(va_acc)
            self.train_hist["prec"].append(tr_p);    self.val_hist["prec"].append(va_p)
            self.train_hist["rec"].append(tr_r);     self.val_hist["rec"].append(va_r)
            self.train_hist["f1"].append(tr_f1);     self.val_hist["f1"].append(va_f1)

            print(f"  Train | Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  P: {tr_p:.4f}  R: {tr_r:.4f}  F1: {tr_f1:.4f}")
            print(f"  Val   | Loss: {va_loss:.4f}  Acc: {va_acc:.4f}  P: {va_p:.4f}  R: {va_r:.4f}  F1: {va_f1:.4f}")
            # CSV logging per epoch
            self._append_csv_log(epoch, tr_loss, tr_acc, tr_p, tr_r, tr_f1, va_loss, va_acc, va_p, va_r, va_f1)

            # Check early stopping
            if self._check_early_stopping(va_loss, va_acc, va_f1, epoch):
                self.early_stopped = True
                print(f"[Early Stopping] Training stopped at epoch {epoch}")
                break

        elapsed = time.perf_counter() - start
        final_epoch = len(self.train_hist['loss'])
        hrs, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        if self.early_stopped:
            print(f"[Early Stopping] Training stopped early at epoch {final_epoch}/{self.epochs}")
        else:
            print(f"Training completed all {final_epoch} epochs")
        print(f"Total training time: {hrs:02d}:{mins:02d}:{secs:02d} (hh:mm:ss)")

        # Optional: auto threshold tuning (binary)
        if self.task_type == "binary" and self.auto_threshold:
            last_val_f1 = self.val_hist["f1"][-1] if self.val_hist["f1"] else -1.0
            best_thr, best_f1 = self.find_best_threshold(steps=self.threshold_search_steps)
            if best_f1 > last_val_f1:
                old_thr = self.threshold
                self.threshold = best_thr
                print(f"[Auto-threshold] Improved F1 {last_val_f1:.4f} → {best_f1:.4f}. "
                      f"Updating threshold: {old_thr:.3f} → {self.threshold:.3f}")
                # Re-evaluate on val for reporting
                va_loss, va_acc, va_p, va_r, va_f1 = self.evaluate(self.val_loader)
                print(f"[Post-threshold] Val | Loss: {va_loss:.4f}  Acc: {va_acc:.4f}  "
                      f"P: {va_p:.4f}  R: {va_r:.4f}  F1: {va_f1:.4f}")
                os.makedirs(self.plots_dir, exist_ok=True)
                with open(os.path.join(self.plots_dir, "best_threshold.txt"), "w") as f:
                    f.write(f"{self.threshold:.6f}\n")
            else:
                print(f"[Auto-threshold] Best searched F1 {best_f1:.4f} did NOT beat last val F1 {last_val_f1:.4f}. "
                      f"Keeping threshold at {self.threshold:.3f}.")

        # Final test metrics (no tuning on test)
        te_loss, te_acc, te_p, te_r, te_f1 = self.evaluate(self.test_loader)
        self.test_metrics = {"loss": te_loss, "acc": te_acc, "prec": te_p, "rec": te_r, "f1": te_f1}
        print(f"[Test] Loss: {te_loss:.4f}  Acc: {te_acc:.4f}  P: {te_p:.4f}  R: {te_r:.4f}  F1: {te_f1:.4f}")

        # If multiclass, also compute confusion matrix on test set
        if self.task_type == "multiclass":
            cm = compute_confusion_matrix(self.model, self.test_loader, self.device, self.num_classes)
            self.test_confusion = cm  # numpy array

    def save(self) -> None:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Always save plain weights (state_dict) for simple loading
        torch.save(self.model.state_dict(), self.weights_path)
        print(f"Weights saved to {self.weights_path}")

        # Also save a rich checkpoint with metadata
        ckpt = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": len(self.train_hist['loss']) if self.early_stopped else self.epochs,
            "early_stopped": getattr(self, "early_stopped", False),
            "best_epoch": getattr(self, "best_epoch", None),
            "best_metric_value": getattr(self, "best_metric_value", None),
            "early_stopping_metric": getattr(self, "early_stopping_metric", None),
            "cfg": self.cfg,
            "task_type": self.task_type,
            "num_classes": self.num_classes,
            "class_names": getattr(self, "class_names", None),
            "input_shape": self.input_shape,             # (C, T)
            "binary_threshold": getattr(self, "threshold", None),
            "train_hist": self.train_hist,
            "val_hist": self.val_hist,
            "test_metrics": self.test_metrics,
        }
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
        torch.save(ckpt, self.ckpt_path)
        print(f"Checkpoint (with metadata) saved to {self.ckpt_path}")

    def _init_csv_log(self) -> None:
        os.makedirs(self.plots_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch",
                    "train_loss","train_acc","train_prec","train_rec","train_f1",
                    "val_loss","val_acc","val_prec","val_rec","val_f1",
                    "lr"
                ])

    def _append_csv_log(
        self, epoch: int,
        tr_loss: float, tr_acc: float, tr_p: float, tr_r: float, tr_f1: float,
        va_loss: float, va_acc: float, va_p: float, va_r: float, va_f1: float
    ) -> None:
        # Try to read LR from first param group; fallback to cfg value
        try:
            lr = float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            lr = float(self.cfg["train"]["learning_rate"])
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                int(epoch),
                float(tr_loss), float(tr_acc), float(tr_p), float(tr_r), float(tr_f1),
                float(va_loss), float(va_acc), float(va_p), float(va_r), float(va_f1),
                lr
            ])

# -----------------------
# Entry
# -----------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("data", {}).get("preprocessed_dir", None) is None:
        raise ValueError("Config must define data.preprocessed_dir")

    task_type = cfg.get("task", {}).get("type", None)
    if task_type not in ("binary", "multiclass"):
        raise ValueError("Config 'task.type' must be 'binary' or 'multiclass'.")

    # sanity check splits
    # Support both val_split/test_split and val_frac/test_frac with sane defaults
    data_cfg = cfg.get("data", {})
    val_frac = float(data_cfg.get("val_split", data_cfg.get("val_frac", 0.1)))
    test_frac = float(data_cfg.get("test_split", data_cfg.get("test_frac", 0.1)))
    if val_frac <= 0 or test_frac <= 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError("Require 0 < val_split, test_split and val_split + test_split < 1.")

    trainer = Trainer(cfg)
    trainer.train()
    trainer.save()
    plot_metrics(trainer.train_hist, trainer.val_hist,
             test_metrics=trainer.test_metrics,
             save_path=trainer.metrics_figure,
             title_prefix="Binary" if trainer.task_type == "binary" else "Macro")

    if trainer.task_type == "multiclass":
        plot_test_confusion_matrix(trainer.test_confusion, class_names=trainer.class_names,
                                normalize=False,
                                save_path=os.path.join(trainer.plots_dir, "test_confusion_matrix.png"))
        plot_test_confusion_matrix(trainer.test_confusion, class_names=trainer.class_names,
                                normalize=True,
                                save_path=os.path.join(trainer.plots_dir, "test_confusion_matrix_norm.png"))



if __name__ == "__main__":
    main()
