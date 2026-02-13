"""
Simplified QAT for hardware constraints:
- Weights: [-1, +1] range with n-bit linear or log2 quantization
- Activations: [0, act_max] range with linear quantization only  
- No batch normalization support (assumes norm: none)
- Weight clipping integrated
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from time import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
from torch.utils.data import DataLoader

from train.utils import (
    _binary_counts, _derive_metrics, _multiclass_confusion_add, _multiclass_macro_prf1,
)
from config import load_config
from model.model import DilatedTCN
from data_loader.utils import make_datasets, get_num_classes
from analysis.plot_metrics import plot_metrics, plot_test_confusion_matrix


def create_qat_dataloaders(cfg, batch_size, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2):
    """Create dataloaders with QAT-specific performance settings"""
    # Update cfg with dataloader settings for make_datasets
    original_batch_size = cfg.get("train", {}).get("batch_size")
    original_num_workers = cfg.get("train", {}).get("num_workers")
    original_pin_memory = cfg.get("train", {}).get("pin_memory")
    original_persistent_workers = cfg.get("train", {}).get("persistent_workers") 
    original_prefetch_factor = cfg.get("train", {}).get("prefetch_factor")
    
    # Temporarily override with QAT settings
    cfg.setdefault("train", {})
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["num_workers"] = num_workers
    cfg["train"]["pin_memory"] = pin_memory
    cfg["train"]["persistent_workers"] = persistent_workers
    cfg["train"]["prefetch_factor"] = prefetch_factor
    
    try:
        train_loader, val_loader, test_loader = make_datasets(cfg, which="all", batch_size=batch_size)
        return train_loader, val_loader, test_loader
    finally:
        # Restore original settings
        if original_batch_size is not None:
            cfg["train"]["batch_size"] = original_batch_size
        if original_num_workers is not None:
            cfg["train"]["num_workers"] = original_num_workers
        if original_pin_memory is not None:
            cfg["train"]["pin_memory"] = original_pin_memory
        if original_persistent_workers is not None:
            cfg["train"]["persistent_workers"] = original_persistent_workers
        if original_prefetch_factor is not None:
            cfg["train"]["prefetch_factor"] = original_prefetch_factor


class HardwareConstrainedQuantizer(nn.Module):
    """Simple quantizer for hardware constraints"""
    
    def __init__(self, method: str = "linear", bits: int = 8, 
                 activation_max: float = 1024.0, log2_levels: int = 4,
                 weight_min: float = -1.0, weight_max: float = 1.0):
        super().__init__()
        self.method = method
        self.bits = bits
        self.activation_max = activation_max
        self.log2_levels = log2_levels
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.enabled = True
        
        # For linear quantization
        if method == "linear":
            self.weight_qmax = (1 << (bits - 1)) - 1  # e.g., 7 for 4-bit
            self.weight_qmin = -self.weight_qmax
            self.weight_scale = self.weight_max / self.weight_qmax  # scale for weight range
        
        # For log2 quantization  
        elif method == "log2":
            # Create power-of-2 levels: ±1, ±2, ±4, ±8, etc.
            self.pos_levels = [2**i for i in range(log2_levels)]  # [1, 2, 4, 8]
            self.neg_levels = [-x for x in reversed(self.pos_levels)]  # [-8, -4, -2, -1]
            self.levels = self.neg_levels + [0] + self.pos_levels
            self.register_buffer("level_tensor", torch.tensor(self.levels, dtype=torch.float32))
    
    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to hardware constraints"""
        if not self.enabled:
            return weights
            
        # First ensure weights are in [weight_min, weight_max] range (should be from clipping)
        w_clipped = torch.clamp(weights, self.weight_min, self.weight_max)
        
        if self.method == "linear":
            # Standard linear quantization in [-1, +1] range
            q_codes = torch.round(w_clipped / self.weight_scale)
            q_codes = torch.clamp(q_codes, self.weight_qmin, self.weight_qmax)
            w_quantized = q_codes * self.weight_scale
            
        elif self.method == "log2":
            # Log2 quantization to power-of-2 levels
            w_quantized = torch.zeros_like(w_clipped)
            for i, val in enumerate(w_clipped.view(-1)):
                if torch.abs(val) < 1e-8:
                    w_quantized.view(-1)[i] = 0.0
                else:
                    # Find closest log2 level
                    distances = torch.abs(self.level_tensor - val)
                    closest_idx = torch.argmin(distances)
                    w_quantized.view(-1)[i] = self.level_tensor[closest_idx]
            
            # Normalize to [weight_min, weight_max] range for hardware
            max_level = max(self.pos_levels)
            w_quantized = w_quantized / max_level * self.weight_max
        
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")
        
        # Straight-through estimator
        return weights + (w_quantized - weights).detach()
    
    def quantize_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Quantize activations to [0, activation_max] range"""
        if not self.enabled:
            return activations
        
        # Ensure activations are non-negative and within bounds
        a_clipped = torch.clamp(activations, 0.0, self.activation_max)
        
        # Linear quantization for activations (always)
        act_qmax = (1 << self.bits) - 1  # unsigned range [0, 2^n-1]
        act_scale = self.activation_max / act_qmax
        
        q_codes = torch.round(a_clipped / act_scale)
        q_codes = torch.clamp(q_codes, 0, act_qmax)
        a_quantized = q_codes * act_scale
        
        # Straight-through estimator 
        return activations + (a_quantized - activations).detach()


def apply_hardware_qat(model: nn.Module, quantizer: HardwareConstrainedQuantizer) -> nn.Module:
    """Apply quantization to model weights and activations"""
    
    # TODO: Add input feature quantization later - temporarily disabled for debugging
    # Add weight quantization to Conv1d and Linear layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            # Store original forward method
            original_forward = module.forward
            
            def create_quantized_forward(orig_forward, quant):
                def quantized_forward(self, x):
                    # Quantize weights
                    if hasattr(self, 'weight'):
                        w_quantized = quant.quantize_weights(self.weight)
                        # Use quantized weight directly without replacing self.weight
                        if isinstance(self, nn.Conv1d):
                            result = F.conv1d(x, w_quantized, self.bias, 
                                            self.stride, self.padding, self.dilation, self.groups)
                        else:  # Linear
                            result = F.linear(x, w_quantized, self.bias)
                        return result
                    else:
                        return orig_forward(x)
                return quantized_forward
            
            # Replace forward method
            module.forward = create_quantized_forward(original_forward, quantizer).__get__(module, type(module))
    
    # Add activation quantization after ReLU layers
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            original_forward = module.forward
            
            def create_quantized_relu(orig_forward, quant):
                def quantized_relu_forward(self, x):
                    x = orig_forward(x)  # Apply ReLU first
                    # Scale to [0, act_max], quantize, then scale back
                    x_scaled = x * quant.activation_max
                    x_quantized = quant.quantize_activations(x_scaled) 
                    return x_quantized / quant.activation_max
                return quantized_relu_forward
            
            module.forward = create_quantized_relu(original_forward, quantizer).__get__(module, type(module))
    
    return model


def clip_model_weights(model: nn.Module, weight_min: float = -1.0, weight_max: float = 1.0) -> None:
    """Clip all model weights to hardware constraints"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                param.clamp_(weight_min, weight_max)


def train_qat_simple(
    model: nn.Module,
    quantizer: HardwareConstrainedQuantizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    task_type: str,
    num_classes: int,
    weight_clipping: bool = True,
    weight_min: float = -1.0,
    weight_max: float = 1.0,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Simple QAT training loop"""
    
    train_hist = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
    val_hist = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        tp = fp = fn = 0
        cm = [[0] * num_classes for _ in range(num_classes)] if task_type == "multiclass" else None
        
        loop = tqdm(train_loader, desc=f"QAT Epoch {epoch}", leave=False)
        for batch_x, batch_y in loop:
            batch_x = batch_x.to(device)
            if task_type == "binary":
                targets = batch_y.float().unsqueeze(1).to(device)
            else:
                targets = batch_y.long().to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            # Apply weight clipping after optimizer step
            if weight_clipping:
                clip_model_weights(model, weight_min, weight_max)
            
            total_loss += loss.item()
            
            # Compute metrics
            if task_type == "binary":
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                correct += (preds == targets.long()).sum().item()
                total += targets.numel()
                _tp, _fp, _fn, _, _ = _binary_counts(preds, targets)
                tp += _tp; fp += _fp; fn += _fn
            else:
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
                cm = _multiclass_confusion_add(cm, preds, targets, num_classes)
        
        # Training metrics
        if task_type == "binary":
            avg_loss, acc, prec, rec, f1 = _derive_metrics(total_loss, len(train_loader), correct, total, tp, fp, fn)
        else:
            avg_loss, acc = _derive_metrics(total_loss, len(train_loader), correct, total)
            prec, rec, f1 = _multiclass_macro_prf1(cm)
        
        train_hist["loss"].append(avg_loss); train_hist["acc"].append(acc)
        train_hist["prec"].append(prec); train_hist["rec"].append(rec); train_hist["f1"].append(f1)
        
        # Validation phase  
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_qat_simple(
            model, val_loader, criterion, device, task_type, num_classes
        )
        val_hist["loss"].append(val_loss); val_hist["acc"].append(val_acc)
        val_hist["prec"].append(val_prec); val_hist["rec"].append(val_rec); val_hist["f1"].append(val_f1)
        
        print(f"[Epoch {epoch}] Train: Loss={avg_loss:.4f} Acc={acc:.4f} F1={f1:.4f} | "
              f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")
        
        # Print weight range verification
        weight_mins, weight_maxs = [], []
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                weight_mins.append(param.min().item())
                weight_maxs.append(param.max().item())
        
        if weight_mins and weight_maxs:
            global_min, global_max = min(weight_mins), max(weight_maxs)
            print(f"[Weights] Range: [{global_min:.4f}, {global_max:.4f}] "
                  f"({'✓' if global_min >= weight_min - 1e-6 and global_max <= weight_max + 1e-6 else '✗'})")
    
    return train_hist, val_hist


def evaluate_qat_simple(
    model: nn.Module,
    loader: DataLoader, 
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
    num_classes: int,
) -> Tuple[float, float, float, float, float]:
    """Simple evaluation for QAT model"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp = fp = fn = 0
    cm = [[0] * num_classes for _ in range(num_classes)] if task_type == "multiclass" else None
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            if task_type == "binary":
                targets = batch_y.float().unsqueeze(1).to(device)
            else:
                targets = batch_y.long().to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            if task_type == "binary":
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                correct += (preds == targets.long()).sum().item()
                total += targets.numel()
                _tp, _fp, _fn, _, _ = _binary_counts(preds, targets)
                tp += _tp; fp += _fp; fn += _fn
            else:
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
                cm = _multiclass_confusion_add(cm, preds, targets, num_classes)
    
    if task_type == "binary":
        return _derive_metrics(total_loss, len(loader), correct, total, tp, fp, fn)
    else:
        avg_loss, acc = _derive_metrics(total_loss, len(loader), correct, total)
        prec, rec, f1 = _multiclass_macro_prf1(cm)
        return avg_loss, acc, prec, rec, f1


def export_hardware_weights(model: nn.Module, quantizer: HardwareConstrainedQuantizer, 
                           out_path: str) -> None:
    """Export quantized weights for hardware deployment"""
    hw_weights = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            # Apply final quantization
            with torch.no_grad():
                w_clipped = torch.clamp(param, quantizer.weight_min, quantizer.weight_max)
                w_quantized = quantizer.quantize_weights(w_clipped)
                hw_weights[name] = w_quantized.cpu().numpy()
    
    np.savez(out_path, **hw_weights)
    print(f"[OK] Exported hardware weights to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple QAT for hardware constraints")
    parser.add_argument("--config", required=True, help="Path to config")
    parser.add_argument("--weights", default=None, help="Pretrained weights to load")
    parser.add_argument("--weight-quantization-method", choices=["linear", "log2"], default=None,
                       help="Weight quantization method (overrides config)")
    parser.add_argument("--weight-linear-quantization-bits", type=int, default=None, help="Weight linear quantization bits (overrides config)")
    parser.add_argument("--weight-log2-quantization-num-levels", type=int, default=None, help="Weight log2 quantization levels (overrides config)")
    parser.add_argument("--weight-min", type=float, default=None, help="Minimum weight value (overrides config)")
    parser.add_argument("--weight-max", type=float, default=None, help="Maximum weight value (overrides config)")
    parser.add_argument("--act-quantization-bits", type=int, default=None, help="Activation quantization bits (overrides config)")
    parser.add_argument("--act-max", type=float, default=None, help="Maximum activation value (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="QAT training epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for QAT (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for QAT (overrides config)")
    
    args = parser.parse_args()
    
    # Load config and setup
    cfg = load_config(args.config)
    qat_cfg = cfg.get("qat", {})
    task_type = cfg["task"]["type"]
    
    # Get QAT parameters from config with command-line overrides
    weight_quantization_method = args.weight_quantization_method if args.weight_quantization_method is not None else qat_cfg.get("weight_quantization_method", "linear")
    weight_linear_quantization_bits = args.weight_linear_quantization_bits if args.weight_linear_quantization_bits is not None else qat_cfg.get("weight_linear_quantization_bits", 4)
    weight_log2_quantization_num_levels = args.weight_log2_quantization_num_levels if args.weight_log2_quantization_num_levels is not None else qat_cfg.get("weight_log2_quantization_num_levels", 4)
    weight_min = args.weight_min if args.weight_min is not None else qat_cfg.get("weight_min", -1.0)
    weight_max = args.weight_max if args.weight_max is not None else qat_cfg.get("weight_max", 1.0)
    act_quantization_bits = args.act_quantization_bits if args.act_quantization_bits is not None else qat_cfg.get("act_quantization_bits", 10)
    act_max = args.act_max if args.act_max is not None else qat_cfg.get("act_max", 1024.0)
    epochs = args.epochs if args.epochs is not None else qat_cfg.get("num_epochs", 5)
    lr = args.lr if args.lr is not None else qat_cfg.get("learning_rate", 1e-4)
    batch_size = args.batch_size if args.batch_size is not None else qat_cfg.get("batch_size", 32)
    
    # Device and dataloader settings
    device_cfg = qat_cfg.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    
    num_workers = qat_cfg.get("num_workers", 2)
    pin_memory = qat_cfg.get("pin_memory", True)
    persistent_workers = qat_cfg.get("persistent_workers", True)
    prefetch_factor = qat_cfg.get("prefetch_factor", 2)
    
    print(f"[QAT] Weight method: {weight_quantization_method}, Weight bits: {weight_linear_quantization_bits}, "
          f"Act bits: {act_quantization_bits}, Act max: {act_max}, Weight range: [{weight_min}, {weight_max}]")
    print(f"[QAT] Device: {device}, Workers: {num_workers}, Pin memory: {pin_memory}")
    
    # Create datasets
    train_loader, val_loader, test_loader = create_qat_dataloaders(
        cfg, batch_size, num_workers, pin_memory, persistent_workers, prefetch_factor
    )
    num_classes = get_num_classes(cfg)
    
    # Create model
    model = DilatedTCN.from_config(cfg)
    if args.weights:
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        print(f"[QAT] Loaded pretrained weights from {args.weights}")
    
    model.to(device)
    
    # Create quantizer
    quantizer = HardwareConstrainedQuantizer(
        method=weight_quantization_method,
        bits=max(weight_linear_quantization_bits, act_quantization_bits), 
        activation_max=act_max,
        log2_levels=weight_log2_quantization_num_levels,
        weight_min=weight_min,
        weight_max=weight_max
    )
    
    # Apply QAT
    model = apply_hardware_qat(model, quantizer)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if task_type == "multiclass" else nn.BCEWithLogitsLoss()
    
    print(f"[QAT] Starting training for {epochs} epochs...")
    train_hist, val_hist = train_qat_simple(
        model, quantizer, train_loader, val_loader, optimizer, criterion,
        device, epochs, task_type, num_classes, weight_clipping=True,
        weight_min=weight_min, weight_max=weight_max
    )
    
    # Test evaluation
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_qat_simple(
        model, test_loader, criterion, device, task_type, num_classes
    )
    print(f"[TEST] Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {test_f1:.4f}")
    
    # Save metrics and plots
    plots_dir = cfg["output"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save training metrics plot
    plot_metrics(train_hist, val_hist, 
                save_path=os.path.join(plots_dir, "qat_metrics.png"),
                title_prefix=f"QAT Training ({weight_quantization_method} {weight_linear_quantization_bits}bit)")
    
    # Save metrics to CSV
    csv_path = os.path.join(plots_dir, "qat_metrics.csv")
    with open(csv_path, 'w') as f:
        # Write header
        f.write("epoch,train_loss,train_acc,train_f1,val_loss,val_acc,val_f1\\n")
        # Write data
        for epoch in range(len(train_hist["loss"])):
            f.write(f"{epoch + 1},{train_hist['loss'][epoch]:.6f},{train_hist['acc'][epoch]:.6f},"
                   f"{train_hist['f1'][epoch]:.6f},{val_hist['loss'][epoch]:.6f},"
                   f"{val_hist['acc'][epoch]:.6f},{val_hist['f1'][epoch]:.6f}\\n")
    print(f"[METRICS] Saved to {csv_path} and {os.path.join(plots_dir, 'qat_metrics.png')}")
    
    # Save results
    output_dir = cfg["output"]["weights_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"qat_simple_{weight_quantization_method}_{weight_linear_quantization_bits}bit.pt")
    torch.save(model.state_dict(), model_path)
    
    weights_path = os.path.join(output_dir, f"qat_simple_{weight_quantization_method}_{weight_linear_quantization_bits}bit.npz")
    export_hardware_weights(model, quantizer, weights_path)
    
    print(f"[DONE] QAT training completed. Model saved to {model_path}")


if __name__ == "__main__":
    main()