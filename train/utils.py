from model.model import DilatedTCN
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import os
from typing import Any, Dict, Optional, Tuple, List, Union, TypeVar

M = TypeVar("M", bound=nn.Module)

# -----------------------
# Loading weights
# -----------------------
def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """
    Load a state_dict from a .pt file. Supports:
      - raw state_dict (param_name -> tensor)
      - checkpoint dict with "model" key
      - checkpoint dict with "state_dict" key
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        print("[INFO] Detected wrapper with 'state_dict' key.")
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        print("[INFO] Detected checkpoint with 'model' key.")
        return obj["model"]
    if isinstance(obj, dict):
        print("[INFO] Detected raw state_dict.")
        return obj
    raise ValueError("Unsupported weights file format: expected a state_dict or checkpoint dict.")


def load_state_dict_forgiving(model: M, path: str, device: torch.device) -> M:
    # Use load_state_dict to handle checkpoint wrappers (state_dict/model keys)
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state_dict = obj["model"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError("Unsupported weights file format: expected a state_dict or checkpoint dict.")
    
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(filtered)
    model.load_state_dict(model_state)
    print(f"[INFO] Loaded weights (forgiving): matched {len(filtered)}/{len(model_state)} tensors")
    return model


# -----------------------
# BatchNorm Folding
# -----------------------
def fold_batchnorm(model: nn.Module) -> nn.Module:
    """
    Fold BatchNorm1d layers into preceding Conv1d layers.
    
    This combines BN parameters (gamma, beta, running_mean, running_var) into
    the Conv weights and biases, then replaces BN with Identity. This is essential
    for hardware deployment and before QAT to match the paper's approach.
    
    Only works with BatchNorm1d. GroupNorm and LayerNorm cannot be folded.
    
    Args:
        model: PyTorch model with Conv1d → BatchNorm1d patterns
    
    Returns:
        Modified model with BN folded into Conv layers
    """
    model.eval()  # Use running stats
    
    # Find all Conv → BN patterns in the model
    modules = list(model.named_modules())
    folded_count = 0
    
    for i, (name, module) in enumerate(modules):
        if isinstance(module, nn.Conv1d):
            # Look for BatchNorm1d immediately following this Conv
            # Check if next module in parent's children is BN
            parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
            conv_attr = name.split(".")[-1]
            
            # Get parent module
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            
            # Check if there's a BN following this conv in the parent's sequential structure
            # This is a heuristic - we look for common patterns like conv1 → norm1
            bn_candidates = []
            if hasattr(parent, f"{conv_attr[:-1]}orm{conv_attr[-1]}"):  # conv1 → norm1
                bn_candidates.append(f"{conv_attr[:-1]}orm{conv_attr[-1]}")
            # Try direct sequence (for nn.Sequential)
            if hasattr(parent, "__getitem__"):
                try:
                    idx = list(parent._modules.keys()).index(conv_attr)
                    if idx + 1 < len(parent._modules):
                        next_key = list(parent._modules.keys())[idx + 1]
                        bn_candidates.append(next_key)
                except (ValueError, AttributeError):
                    pass
            
            for bn_attr in bn_candidates:
                if hasattr(parent, bn_attr):
                    bn = getattr(parent, bn_attr)
                    if isinstance(bn, nn.BatchNorm1d):
                        # Fold BN into Conv
                        conv = module
                        _fold_bn_into_conv(conv, bn)
                        # Replace BN with Identity
                        setattr(parent, bn_attr, nn.Identity())
                        folded_count += 1
                        print(f"[BN-FOLD] {name} → {parent_name}.{bn_attr if parent_name else bn_attr}")
                        break
    
    print(f"[OK] Folded {folded_count} BatchNorm layers into Conv layers")
    return model


def _fold_bn_into_conv(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> None:
    """
    Fold BatchNorm parameters into Conv1d weights and bias.
    
    Math:
        BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
        Folded: w_new = w * gamma / sqrt(var + eps)
                b_new = (b - mean) * gamma / sqrt(var + eps) + beta
    
    Args:
        conv: Conv1d layer to modify in-place
        bn: BatchNorm1d layer to fold
    """
    # Get BN parameters
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    
    # Compute scaling factor: gamma / sqrt(var + eps)
    std = torch.sqrt(var + eps)
    scale = gamma / std
    
    # Fold into conv weights
    # Conv weight shape: (out_channels, in_channels, kernel_size)
    # Scale each output channel
    conv.weight.data = conv.weight.data * scale.view(-1, 1, 1)
    
    # Fold into conv bias
    if conv.bias is None:
        # Create bias if it doesn't exist
        conv.bias = nn.Parameter(torch.zeros(conv.out_channels, device=conv.weight.device))
    
    conv.bias.data = (conv.bias.data - mean) * scale + beta


# -----------------------
# Metrics helpers
# -----------------------
def _binary_counts(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int, int, int, int]:
    preds = preds.long()
    targets = targets.long()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return tp, fp, fn, correct, total

def _derive_metrics(
    total_loss: float,
    num_batches: int,
    correct: int,
    total: int,
    tp: Optional[int] = None,
    fp: Optional[int] = None,
    fn: Optional[int] = None,
) -> Union[Tuple[float, float], Tuple[float, float, float, float, float]]:
    avg_loss = total_loss / max(1, num_batches)
    accuracy = correct / total if total > 0 else 0.0
    if tp is None:
        return avg_loss, accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return avg_loss, accuracy, precision, recall, f1

def _multiclass_confusion_add(cm: List[List[int]], preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> List[List[int]]:
    for p, t in zip(preds.tolist(), targets.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm

def _multiclass_macro_prf1(cm: List[List[int]]) -> Tuple[float, float, float]:
    K = len(cm)
    precisions, recalls, f1s = [], [], []
    for k in range(K):
        tp = cm[k][k]
        fp = sum(cm[r][k] for r in range(K)) - tp
        fn = sum(cm[k][c] for c in range(K)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    macro_p = sum(precisions) / K if K > 0 else 0.0
    macro_r = sum(recalls) / K if K > 0 else 0.0
    macro_f1 = sum(f1s) / K if K > 0 else 0.0
    return macro_p, macro_r, macro_f1

def compute_confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> np.ndarray:
    """Return confusion matrix (num_classes x num_classes) of counts for multiclass."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().cpu().numpy()
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for t, p in zip(y, preds):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    cm[t, p] += 1
    return cm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task_type: str,
    threshold: float = 0.5,
    num_classes: Optional[int] = None
) -> Tuple[float, float, float, float]:
    """
    Evaluate model on a dataset and return accuracy, precision, recall, F1.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with (features, labels)
        device: Device to run evaluation on
        task_type: 'binary' or 'multiclass'
        threshold: Classification threshold for binary task
        num_classes: Number of classes for multiclass task
    
    Returns:
        (accuracy, precision, recall, f1)
    """
    model.eval().to(device)
    total_loss, correct, total, tp, fp, fn = 0.0, 0, 0, 0, 0, 0
    cm = [[0] * num_classes for _ in range(num_classes)] if task_type == "multiclass" else None
    criterion = nn.BCEWithLogitsLoss() if task_type == "binary" else nn.CrossEntropyLoss()
    
    for x, y in dataloader:
        x = x.to(device)
        targets = y.float().unsqueeze(1).to(device) if task_type == "binary" else y.long().to(device)
        logits = model(x)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        
        if task_type == "binary":
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            correct += (preds == targets.long()).sum().item()
            total += targets.numel()
            _tp, _fp, _fn, _, _ = _binary_counts(preds, targets)
            tp += _tp
            fp += _fp
            fn += _fn
        else:
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
            cm = _multiclass_confusion_add(cm, preds, targets, num_classes)
    
    if task_type == "binary":
        _, acc, prec, rec, f1 = _derive_metrics(total_loss, len(dataloader), correct, total, tp, fp, fn)
    else:
        _, acc = _derive_metrics(total_loss, len(dataloader), correct, total)
        prec, rec, f1 = _multiclass_macro_prf1(cm)
    
    return acc, prec, rec, f1
