
"""
Simplified QAT for hardware constraints:
- Weights: [-1, +1] range with n-bit linear or log2 quantization
- Activations: [0, act_max] range with linear quantization only
- No batch normalization support (assumes norm: none)
- Weight clipping integrated
"""

import os
import argparse
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from train.utils import (
    _binary_counts, _derive_metrics, _multiclass_confusion_add, _multiclass_macro_prf1,
)
from config import load_config
from model.model import DilatedTCN
from data_loader.utils import make_datasets, get_num_classes
from analysis.plot_metrics import plot_metrics
from quantization.core import quantize_tensor, qmax_for_bits, compute_log2_levels, quantize_to_log2


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
    """Hardware quantizer using shared quantization.core functions for consistency with PTQ"""

    def __init__(self, method: str = "linear", weight_bits: int = 8, activation_bits: int = 10,
                 activation_min: float = 0.0, activation_max: float = 1024.0, log2_levels: int = 4,
                 weight_min: float = -1.0, weight_max: float = 1.0, normalize_to_unit: bool = False):
        super().__init__()
        self.method = method
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.activation_min = activation_min
        self.activation_max = activation_max
        self.log2_levels = log2_levels
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.normalize_to_unit = normalize_to_unit
        self.enabled = True

        # Global discrete quantization levels (computed later)
        self.quantization_levels = None
        self.lsb = None

        # For linear quantization - use discrete levels instead of scale
        if method == "linear":
            self.weight_qmax = (1 << (weight_bits - 1)) - 1  # e.g., 7 for 4-bit
            self.weight_qmin = -self.weight_qmax
        # For log2 quantization: levels will be computed dynamically
    
    def compute_global_levels(self, model: nn.Module) -> None:
        """Compute quantization parameters using quantization.core functions"""
        if self.method == "linear":
            num_levels = 2 ** self.weight_bits
            self.qmax = qmax_for_bits(self.weight_bits)
            self.qmin = -self.qmax
            self.quantization_levels = np.linspace(self.weight_min, self.weight_max, num_levels, dtype=np.float32)
            self.lsb = (self.weight_max - self.weight_min) / (num_levels - 1)
            print(f"[QAT] Global quantization: {num_levels} discrete levels")
            print(f"[QAT] Levels: [{self.weight_min:.6f} ... {self.weight_max:.6f}], LSB: {self.lsb:.6f}")
            print(f"[QAT] First few levels: {self.quantization_levels[:5]}")
            print(f"[QAT] Last few levels: {self.quantization_levels[-5:]}")
        elif self.method == "log2":
            # Will compute levels dynamically based on data
            pass
    
    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights using quantization.core functions"""
        if not self.enabled:
            return weights
        w_clipped = torch.clamp(weights, self.weight_min, self.weight_max)
        if self.method == "linear":
            # Use symmetric quantization (zero_point=0)
            scale = (self.weight_max - self.weight_min) / (2 * self.qmax)  # assumes symmetric grid
            w_quantized = quantize_tensor(w_clipped, scale, 0, self.qmin, self.qmax, self.normalize_to_unit)
        elif self.method == "log2":
            w_np = w_clipped.cpu().numpy()
            max_abs = np.abs(w_np).max()
            w_quantized_np = quantize_to_log2(w_np, self.log2_levels, max_abs, normalize_to_unit=self.normalize_to_unit)
            w_quantized = torch.from_numpy(w_quantized_np).to(weights.device)
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")
        # Straight-through estimator
        return weights + (w_quantized - weights).detach()
    
    def quantize_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Quantize activations to [activation_min, activation_max] range"""
        if not self.enabled:
            return activations
        
        # Ensure activations are within bounds
        a_clipped = torch.clamp(activations, self.activation_min, self.activation_max)
        
        # Linear quantization for activations (always)
        act_qmax = (1 << self.activation_bits) - 1  # unsigned range [0, 2^n-1]
        act_range = self.activation_max - self.activation_min
        act_scale = act_range / act_qmax
        
        q_codes = torch.round((a_clipped - self.activation_min) / act_scale)
        q_codes = torch.clamp(q_codes, 0, act_qmax)
        a_quantized = q_codes * act_scale + self.activation_min
        
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
            
            # Create quantized forward function with closure over module and quantizer
            def create_quantized_forward(target_module, quant_func):
                def quantized_forward(x):
                    # Quantize weights using the specific module and quantizer
                    if hasattr(target_module, 'weight'):
                        w_quantized = quant_func.quantize_weights(target_module.weight)
                        # Use quantized weight directly without replacing self.weight
                        if isinstance(target_module, nn.Conv1d):
                            result = F.conv1d(x, w_quantized, target_module.bias, 
                                            target_module.stride, target_module.padding, 
                                            target_module.dilation, target_module.groups)
                        else:  # Linear
                            result = F.linear(x, w_quantized, target_module.bias)
                        return result
                    else:
                        return original_forward(x)
                return quantized_forward
            
            # Replace forward method with simpler binding
            module.forward = create_quantized_forward(module, quantizer)
    
    # Add activation quantization after ReLU layers
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            # Create quantized forward function with closure over module and quantizer (like weight quantization)
            def create_quantized_relu(target_module, quant_func):
                def quantized_relu_forward(x):
                    x = torch.relu(x)  # Apply ReLU directly (equivalent to original forward)
                    # Directly quantize activations (they're already non-negative from ReLU)
                    x_quantized = quant_func.quantize_activations(x)
                    return x_quantized
                return quantized_relu_forward
            
            # Replace forward method with consistent pattern (like weight quantization)
            module.forward = create_quantized_relu(module, quantizer)
    
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
    act_min: float = 0.0,
    act_max: float = 1024.0,
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
        
        # Validation phase (with hardware constraints for consistent evaluation)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_qat_simple(
            model, val_loader, criterion, device, task_type, num_classes, 
            apply_constraints=True, act_min=act_min, act_max=act_max
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
    apply_constraints: bool = True,
    act_min: float = 0.0,
    act_max: float = 1024.0
) -> Tuple[float, float, float, float, float]:
    """Evaluation for QAT model with consistent constraint application matching PTQ"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp = fp = fn = 0
    cm = [[0] * num_classes for _ in range(num_classes)] if task_type == "multiclass" else None
    
    # Apply activation constraint hooks if requested (matches PTQ evaluation)
    hooks = []
    if apply_constraints:
        def activation_constraint_hook(module, input, output):
            return torch.clamp(output, act_min, act_max)
        
        # Apply hooks to modules with 'act' in name (matches PTQ evaluation)
        for name, module in model.named_modules():
            if 'act' in name:  # Apply to activation modules by name
                hooks.append(module.register_forward_hook(activation_constraint_hook))
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            if task_type == "binary":
                targets = batch_y.float().unsqueeze(1).to(device)
            else:
                targets = batch_y.long().to(device)
            
            # Apply per-sample input scaling for hardware constraints (matches PTQ)
            if apply_constraints:
                # Per-sample normalization to [act_min, act_max] using config limits
                # Handles 2D (batch_size, features) and 3D (batch_size, features, time)
                dims = tuple(range(1, batch_x.dim()))
                min_vals = batch_x.amin(dim=dims, keepdim=True)
                max_vals = batch_x.amax(dim=dims, keepdim=True)
                denom = (max_vals - min_vals).clamp(min=1e-8)
                # Scale to config activation range per input (same as PTQ)
                batch_x = (batch_x - min_vals) / denom * (act_max - act_min) + act_min
            
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
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    if task_type == "binary":
        return _derive_metrics(total_loss, len(loader), correct, total, tp, fp, fn)
    else:
        avg_loss, acc = _derive_metrics(total_loss, len(loader), correct, total)
        prec, rec, f1 = _multiclass_macro_prf1(cm)
        return avg_loss, acc, prec, rec, f1


def export_hardware_weights(model: nn.Module, quantizer: HardwareConstrainedQuantizer, 
                           out_path: str) -> None:
    """Export quantized weights for hardware deployment (weights already quantized during training)"""
    hw_weights = {}
    
    # Apply final quantization to convert continuous training parameters to discrete hardware weights
    # Straight-through estimator keeps parameters continuous - need final quantization for hardware
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            with torch.no_grad():
                # Apply the same quantization used during forward pass to get discrete values
                w_clipped = torch.clamp(param.data, quantizer.weight_min, quantizer.weight_max)
                w_quantized = quantizer.quantize_weights(w_clipped)
                hw_weights[name] = w_quantized.detach().cpu().numpy()
    
    np.savez(out_path, **hw_weights)
    print(f"[OK] Exported hardware weights to {out_path}")
    
    # Report quantization stats
    total_unique = set()
    for name, weights in hw_weights.items():
        unique_vals = np.unique(weights)
        total_unique.update(unique_vals)
        print(f"  {name}: {len(unique_vals)} unique levels, range [{weights.min():.6f}, {weights.max():.6f}]")
    
    print(f"[GLOBAL] Total unique values across all layers: {len(total_unique)}")
    if quantizer.method == "linear" and hasattr(quantizer, 'quantization_levels'):
        expected_levels = len(quantizer.quantization_levels)
        print(f"[GLOBAL] Expected {expected_levels} levels for {quantizer.weight_bits}-bit quantization")
        if len(total_unique) == expected_levels:
            print("[GLOBAL] ✓ Quantization successful - exact discrete levels achieved")
        else:
            print(f"[GLOBAL] ✗ Quantization issue - got {len(total_unique)} instead of {expected_levels} levels")


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
    act_min = qat_cfg.get("act_min", 0.0)  # Get from config for consistency with PTQ
    act_max = args.act_max if args.act_max is not None else qat_cfg.get("act_max", 1024.0)
    epochs = args.epochs if args.epochs is not None else qat_cfg.get("num_epochs", 5)
    lr = args.lr if args.lr is not None else qat_cfg.get("learning_rate", 1e-4)
    batch_size = args.batch_size if args.batch_size is not None else qat_cfg.get("batch_size", 32)
    normalize_to_unit = qat_cfg.get("normalize_to_unit", False)  # Get from config for consistency with PTQ
    
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
          f"Act bits: {act_quantization_bits}, Act range: [{act_min}, {act_max}], Weight range: [{weight_min}, {weight_max}]")
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
        weight_bits=weight_linear_quantization_bits,
        activation_bits=act_quantization_bits,
        activation_min=act_min,
        activation_max=act_max,
        log2_levels=weight_log2_quantization_num_levels,
        weight_min=weight_min,
        weight_max=weight_max,
        normalize_to_unit=normalize_to_unit
    )
    
    # Compute global quantization levels before applying QAT
    quantizer.compute_global_levels(model)
    
    # Apply QAT
    model = apply_hardware_qat(model, quantizer)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if task_type == "multiclass" else nn.BCEWithLogitsLoss()
    
    # Evaluate initial quantized model (before any QAT training)
    print(f"[QAT] Evaluating initial quantized model on validation set...")
    initial_val_loss, initial_val_acc, initial_val_prec, initial_val_rec, initial_val_f1 = evaluate_qat_simple(
        model, val_loader, criterion, device, task_type, num_classes, 
        apply_constraints=True, act_min=act_min, act_max=act_max
    )
    print(f"[Initial] Val: Loss={initial_val_loss:.4f} Acc={initial_val_acc:.4f} "
          f"Prec={initial_val_prec:.4f} Rec={initial_val_rec:.4f} F1={initial_val_f1:.4f}")
    
    print(f"[QAT] Starting training for {epochs} epochs...")
    train_hist, val_hist = train_qat_simple(
        model, quantizer, train_loader, val_loader, optimizer, criterion,
        device, epochs, task_type, num_classes, weight_clipping=True,
        weight_min=weight_min, weight_max=weight_max, act_min=act_min, act_max=act_max
    )
    
    # Test evaluation (with hardware constraints for consistent evaluation)
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_qat_simple(
        model, test_loader, criterion, device, task_type, num_classes, 
        apply_constraints=True, act_min=act_min, act_max=act_max
    )
    print(f"[TEST] Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {test_f1:.4f}")
    
    # Save metrics and plots
    plots_dir = os.path.join(cfg["output"]["plots_dir"], "qat/training")
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
    output_dir = os.path.join(cfg["output"]["weights_dir"], "qat")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"qat_weights_{weight_quantization_method}_{weight_linear_quantization_bits}bit_act_{act_quantization_bits}bit.pt")
    torch.save(model.state_dict(), model_path)
    
    weights_path = os.path.join(output_dir, f"qat_weights_{weight_quantization_method}_{weight_linear_quantization_bits}bit_act_{act_quantization_bits}bit.npz")
    export_hardware_weights(model, quantizer, weights_path)
    
    print(f"[DONE] QAT training completed. Model saved to {model_path}")


if __name__ == "__main__":
    main()