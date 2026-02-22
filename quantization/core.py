

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from copy import deepcopy

def qmax_for_bits(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def compute_log2_levels(max_abs_val: float, num_levels: int = 8) -> tuple[np.ndarray, float]:
    """
    Compute fixed log2 quantization levels and scale factor for hardware implementation.
    
    Args:
        max_abs_val: Maximum absolute value in the data (typically percentile value)
        num_levels: Number of levels determines max level as 2^num_levels
    
    Returns:
        Tuple of (fixed_levels_array, scale_factor)
        - levels: Fixed array [-2^num_levels, ..., -1, 0, 1, ..., 2^num_levels]
        - scale_factor: Factor to scale weights so max_abs_val maps to 2^num_levels
    """
    if max_abs_val == 0:
        levels = np.array([0])
        return levels, 1.0
    
    # Create fixed power-of-2 levels: ±1, ±2, ±4, ±8, ... up to num_levels
    pos_levels = np.array([2**i for i in range(num_levels)])  # [1, 2, 4, 8] for num_levels=4
    
    # Full level set: negative levels, zero, positive levels
    neg_levels = -pos_levels[::-1]  # [-8, -4, -2, -1] for num_levels=4
    levels = np.concatenate([neg_levels, [0], pos_levels])  # [-8,-4,-2,-1,0,1,2,4,8]
    
    # Compute scale factor: scale weights so max_abs_val maps to max level
    max_level = pos_levels[-1]  # Maximum positive level (e.g., 8 for num_levels=4)
    scale_factor = max_abs_val / max_level
    
    return levels, scale_factor


def quantize_to_log2(data: np.ndarray, num_levels: int = 8, max_abs_val: Optional[float] = None, 
                    normalize_to_unit: bool = False) -> np.ndarray:
    """
    Quantize data to fixed log2 levels for hardware implementation.
    
    Args:
        data: Input data to quantize
        num_levels: Number of levels determines available powers of 2: [1, 2, 4, ..., 2^(num_levels-1)]
        max_abs_val: Maximum absolute value for scaling (if None, use data max)
        normalize_to_unit: If True, normalize output to [-1, +1] range for hardware
    
    Returns:
        Quantized data in original scale (or [-1,+1] if normalize_to_unit=True)
    """
    if data.size == 0:
        return data.copy().astype(np.float32)
    
    if max_abs_val is None:
        max_abs_val = np.abs(data).max()
    
    if max_abs_val == 0:
        return np.zeros_like(data, dtype=np.float32)
    
    # Get fixed levels and scale factor
    levels, scale_factor = compute_log2_levels(max_abs_val, num_levels)
    
    # Get the maximum level for normalization
    max_level = 2 ** (num_levels - 1)  # e.g., for num_levels=4: 2^3 = 8
    
    # Scale data to fit the fixed levels
    scaled_data = data / scale_factor
    
    # Quantize to nearest fixed level
    quantized = np.zeros_like(scaled_data, dtype=np.float32)
    for i, val in enumerate(scaled_data.flat):
        if val == 0:
            quantized.flat[i] = 0.0
        else:
            # Find closest level by minimizing absolute difference
            distances = np.abs(levels - val)
            closest_idx = np.argmin(distances)
            closest_level = levels[closest_idx]
            
            if normalize_to_unit:
                # Normalize to [-1, +1] range for hardware deployment
                quantized.flat[i] = closest_level / max_level
            else:
                # Scale back to original magnitude
                quantized.flat[i] = closest_level * scale_factor
    
    return quantized


def quantize_array_to_log2_codes(data: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    Quantize numpy array to log2 level codes (integer indices).
    
    Args:
        data: Input data array (should already be scaled to level range)
        levels: Array of fixed log2 levels (e.g., [-8, -4, -2, -1, 0, 1, 2, 4, 8])
    
    Returns:
        Integer codes corresponding to level indices
    """
    if data.size == 0:
        return np.array([], dtype=np.int32)
    
    codes = np.zeros_like(data, dtype=np.int32)
    zero_idx = len(levels) // 2  # Index of zero level
    
    for i, val in enumerate(data.flat):
        # Find closest level index
        distances = np.abs(levels - val)
        closest_idx = np.argmin(distances)
        codes.flat[i] = closest_idx - zero_idx  # Center around 0 for symmetric codes
    
    return codes


def compute_scale_symmetric(data: np.ndarray, qmax: float, percentile: float = 100.0) -> float:
    """Compute symmetric quantization scale from data."""
    max_abs = np.percentile(np.abs(data), percentile)
    return max_abs / qmax if max_abs > 0 else 1.0


def compute_scale_asymmetric(data: np.ndarray, qmin: float, qmax: float, percentile: float = 100.0) -> tuple:
    """Compute asymmetric quantization scale and zero point from data."""
    min_val = np.percentile(data, 100 - percentile)
    max_val = np.percentile(data, percentile)
    scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
    zero_point = np.round(qmin - min_val / scale)
    return scale, zero_point


def quantize_tensor(data: torch.Tensor, scale: float, zero_point: float, qmin: float, qmax: float, 
                    normalize_to_unit: bool = False) -> torch.Tensor:
    """Quantize and dequantize a tensor."""
    if zero_point == 0:
        # Symmetric
        q = torch.clamp(torch.round(data / scale), qmin, qmax)
        if normalize_to_unit:
            # Normalize to [-1, +1] range for hardware deployment
            return q / qmax
        else:
            # Return in original scale
            return q * scale
    else:
        # Asymmetric
        q = torch.clamp(torch.round(data / scale + zero_point), qmin, qmax)
        if normalize_to_unit:
            # Normalize to [-1, +1] range (asymmetric case more complex)
            q_centered = q - zero_point
            return q_centered / max(abs(qmin), abs(qmax))
        else:
            return (q - zero_point) * scale


def quantize_array_to_codes(data: np.ndarray, scale: float, qmin: float, qmax: float) -> np.ndarray:
    """Quantize numpy array to integer codes (symmetric only)."""
    if scale == 0 or np.abs(data).max() == 0:
        return np.zeros_like(data, dtype=np.int32)
    q = np.round(data / scale).clip(qmin, qmax)
    return q.astype(np.int32)


def quantize_weights_by_scheme(
    state_dict: Dict[str, torch.Tensor],
    bits: int,
    scheme: str = "per_tensor",
    global_percentile: float = 100.0,
    return_codes: bool = False,
    method: str = "linear",
    num_log2_levels: int = 8,
    normalize_to_unit: bool = False,
) -> np.ndarray:
    """
    Quantize all .weight tensors in a state_dict according to the specified scheme.
    
    Args:
        state_dict: Dictionary of parameter tensors
        bits: Bitwidth for quantization (ignored for log2 method)
        scheme: One of "per_channel", "per_tensor", "global"
        global_percentile: Percentile for global scheme scale computation
        return_codes: If True, return concatenated integer codes; if False, return scales
        method: Quantization method - "linear" (default) or "log2"
        num_log2_levels: Number of positive/negative levels for log2 quantization
        normalize_to_unit: If True, normalize quantized weights to [-1, +1] range
        
    Returns:
        Concatenated numpy array of quantized integer codes (if return_codes=True)
    """
    if method == "log2":
        return _quantize_weights_log2_scheme(
            state_dict, scheme, global_percentile, return_codes, num_log2_levels, normalize_to_unit
        )
    
    # Original linear quantization
    qmax = qmax_for_bits(bits)
    codes = []
    
    if scheme == "global":
        # Collect all weights for global scale
        all_weights = []
        for name, t in state_dict.items():
            if isinstance(t, torch.Tensor) and t.dtype.is_floating_point and name.endswith(".weight"):
                all_weights.append(t.cpu().numpy().ravel())
        if not all_weights:
            return np.array([], dtype=np.int32)
        all_weights_arr = np.concatenate(all_weights, axis=0)
        scale = compute_scale_symmetric(all_weights_arr, qmax, global_percentile)
        return quantize_array_to_codes(all_weights_arr, scale, -qmax, qmax) if return_codes else scale
    
    elif scheme == "per_tensor":
        for name, t in state_dict.items():
            if not isinstance(t, torch.Tensor) or not t.dtype.is_floating_point:
                continue
            if not name.endswith(".weight"):
                continue
            w = t.cpu().numpy().ravel()
            scale = compute_scale_symmetric(w, qmax)
            if return_codes:
                q = quantize_array_to_codes(w, scale, -qmax, qmax)
                codes.append(q)
        if not codes:
            return np.array([], dtype=np.int32)
        return np.concatenate(codes, axis=0) if return_codes else None
    
    else:  # per_channel
        for name, t in state_dict.items():
            if not isinstance(t, torch.Tensor) or not t.dtype.is_floating_point:
                continue
            if not name.endswith(".weight"):
                continue
            w = t.cpu().numpy()
            if w.ndim < 2:
                # fallback to per-tensor
                w_flat = w.ravel()
                scale = compute_scale_symmetric(w_flat, qmax)
                if return_codes:
                    q = quantize_array_to_codes(w_flat, scale, -qmax, qmax)
                    codes.append(q)
                continue
            # per-channel quantization
            oc = w.shape[0]
            w_flat = w.reshape(oc, -1)
            for ch_data in w_flat:
                scale = compute_scale_symmetric(ch_data, qmax)
                if return_codes:
                    q = quantize_array_to_codes(ch_data, scale, -qmax, qmax)
                    codes.append(q)
        if not codes:
            return np.array([], dtype=np.int32)
        return np.concatenate(codes, axis=0) if return_codes else None


def _quantize_weights_log2_scheme(
    state_dict: Dict[str, torch.Tensor],
    scheme: str,
    global_percentile: float,
    return_codes: bool,
    num_log2_levels: int,
    normalize_to_unit: bool = False,
) -> np.ndarray:
    """Helper function for log2 quantization schemes."""
    codes = []
    
    if scheme == "global":
        # Collect all weights for global log2 levels
        all_weights = []
        for name, t in state_dict.items():
            if isinstance(t, torch.Tensor) and t.dtype.is_floating_point and name.endswith(".weight"):
                all_weights.append(t.cpu().numpy().ravel())
        if not all_weights:
            return np.array([], dtype=np.int32)
        all_weights_arr = np.concatenate(all_weights, axis=0)
        max_abs = np.percentile(np.abs(all_weights_arr), global_percentile)
        levels, scale_factor = compute_log2_levels(max_abs, num_log2_levels)
        quantized = quantize_to_log2(all_weights_arr, num_log2_levels, max_abs, normalize_to_unit)
        return quantized if return_codes else levels  # quantized is already integer codes
    
    elif scheme == "per_tensor":
        for name, t in state_dict.items():
            if not isinstance(t, torch.Tensor) or not t.dtype.is_floating_point:
                continue
            if not name.endswith(".weight"):
                continue
            w = t.cpu().numpy().ravel()
            if return_codes:
                max_abs = np.abs(w).max()
                quantized = quantize_to_log2(w, num_log2_levels, max_abs, normalize_to_unit)
                codes.append(quantized)  # quantized is already integer codes
        if not codes:
            return np.array([], dtype=np.int32)
        return np.concatenate(codes, axis=0) if return_codes else None
    
    else:  # per_channel
        for name, t in state_dict.items():
            if not isinstance(t, torch.Tensor) or not t.dtype.is_floating_point:
                continue
            if not name.endswith(".weight"):
                continue
            w = t.cpu().numpy()
            if w.ndim < 2:
                # fallback to per-tensor
                w_flat = w.ravel()
                if return_codes:
                    max_abs = np.abs(w_flat).max()
                    quantized = quantize_to_log2(w_flat, num_log2_levels, max_abs, normalize_to_unit)
                    codes.append(quantized)  # quantized is already integer codes
                continue
            # per-channel quantization
            oc = w.shape[0]
            w_flat = w.reshape(oc, -1)
            for ch_data in w_flat:
                if return_codes:
                    max_abs = np.abs(ch_data).max()
                    quantized = quantize_to_log2(ch_data, num_log2_levels, max_abs, normalize_to_unit)
                    codes.append(quantized)  # quantized is already integer codes
        if not codes:
            return np.array([], dtype=np.int32)
        return np.concatenate(codes, axis=0) if return_codes else None


def _quantize_model_weights_log2(
    model: torch.nn.Module,
    num_log2_levels: int,
    scheme: str,
    global_percentile: float,
    normalize_to_unit: bool
) -> torch.nn.Module:
    """
    Helper function to apply log2 quantization to model weights.
    
    Args:
        model: Model to quantize (already on CPU)
        num_log2_levels: Number of positive/negative levels for log2 quantization
        scheme: One of "per_tensor", "per_channel", "global"
        global_percentile: Percentile for global scale computation
        normalize_to_unit: If True, normalize to [-1,+1] range
        
    Returns:
        Quantized model (modified in-place)
    """
    if scheme == "global":
        # Collect all weights for global log2 scale computation
        all_weights = []
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                all_weights.append(module.weight.data.cpu().numpy().flatten())
        
        if all_weights:
            all_weights_arr = np.concatenate(all_weights)
            global_max = np.percentile(np.abs(all_weights_arr), global_percentile)
            
            # Apply global log2 quantization to each layer
            for module in model.modules():
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                    w_np = module.weight.data.cpu().numpy()
                    w_quantized = quantize_to_log2(w_np, num_log2_levels, global_max, normalize_to_unit)
                    module.weight.data = torch.from_numpy(w_quantized).float()
    
    elif scheme == "per_tensor":
        # Per-tensor log2 quantization
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                w_np = module.weight.data.cpu().numpy()
                max_abs = np.abs(w_np).max()
                w_quantized = quantize_to_log2(w_np, num_log2_levels, max_abs, normalize_to_unit)
                module.weight.data = torch.from_numpy(w_quantized).float()
    
    elif scheme == "per_channel":
        # Per-channel log2 quantization
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                w = module.weight.data
                if w.dim() < 2:
                    # Fallback to per-tensor for 1D weights
                    w_np = w.cpu().numpy()
                    max_abs = np.abs(w_np).max()
                    w_quantized = quantize_to_log2(w_np, num_log2_levels, max_abs, normalize_to_unit)
                    module.weight.data = torch.from_numpy(w_quantized).float()
                    continue
                
                # Per-channel quantization
                oc = w.shape[0]
                w_np = w.cpu().numpy()
                w_flat = w_np.reshape(oc, -1)
                
                # Quantize each channel separately
                quantized_channels = []
                for ch_data in w_flat:
                    max_abs = np.abs(ch_data).max()
                    ch_quantized = quantize_to_log2(ch_data, num_log2_levels, max_abs, normalize_to_unit)
                    quantized_channels.append(ch_quantized)
                
                w_quantized = np.stack(quantized_channels).reshape(w_np.shape)
                module.weight.data = torch.from_numpy(w_quantized).float()
    
    return model


def quantize_model_weights(
    model: torch.nn.Module,
    bits: int = 8,
    scheme: str = "per_tensor",
    symmetric: bool = True,
    global_percentile: float = 100.0,
    normalize_to_unit: bool = False,
    method: str = "linear",
    num_log2_levels: int = 8
) -> torch.nn.Module:
    """
    Quantize weights of all Conv1d/Linear layers in the model using the specified scheme.
    
    Args:
        model: Model to quantize
        bits: Bitwidth for linear quantization (ignored for log2 method)
        scheme: One of "per_tensor", "per_channel", "global"
        symmetric: Use symmetric quantization (only for linear method)
        global_percentile: Percentile for global scale computation
        normalize_to_unit: If True, normalize quantized weights to [-1,+1] range for hardware
        method: Quantization method - "linear" (default) or "log2"
        num_log2_levels: Number of positive/negative levels for log2 quantization
    
    Returns:
        Quantized model (modified in-place and returned)
    """
    model = model.cpu()
    
    # Route to appropriate quantization method
    if method == "log2":
        return _quantize_model_weights_log2(model, num_log2_levels, scheme, global_percentile, normalize_to_unit)
    
    # Linear quantization (original implementation)
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax_w = (2 ** (bits - 1)) - 1
    else:
        qmin = 0
        qmax_w = (2 ** bits) - 1

    if scheme == "global":
        all_weights = []
        for _, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                w = module.weight.data
                all_weights.append(w.view(-1))
        if all_weights:
            all_weights_np = torch.cat(all_weights).cpu().numpy()
            if symmetric:
                scale = compute_scale_symmetric(all_weights_np, qmax_w, global_percentile)
                zero_point = 0
            else:
                scale, zero_point = compute_scale_asymmetric(all_weights_np, qmin, qmax_w, global_percentile)
            for _, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                    module.weight.data = quantize_tensor(module.weight.data, scale, zero_point, qmin, qmax_w, normalize_to_unit)
    elif scheme == "per_channel":
        for _, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                w = module.weight.data
                if w.dim() < 2:
                    # Fallback to per-tensor
                    w_np = w.cpu().numpy().ravel()
                    if symmetric:
                        scale = compute_scale_symmetric(w_np, qmax_w)
                        zero_point = 0
                    else:
                        scale, zero_point = compute_scale_asymmetric(w_np, qmin, qmax_w)
                    module.weight.data = quantize_tensor(w, scale, zero_point, qmin, qmax_w, normalize_to_unit)
                    continue
                oc = w.shape[0]
                w_flat = w.view(oc, -1)
                if symmetric:
                    max_abs = w_flat.abs().max(dim=1).values
                    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax_w)
                    zero_point = torch.zeros_like(scale)
                    q = torch.clamp(torch.round(w_flat / scale[:, None]), qmin, qmax_w)
                    module.weight.data = (q * scale[:, None]).view_as(w)
                else:
                    min_val = w_flat.min(dim=1).values
                    max_val = w_flat.max(dim=1).values
                    scale = torch.where(max_val > min_val, (max_val - min_val) / (qmax_w - qmin), torch.ones_like(max_val))
                    zero_point = torch.round(qmin - min_val / scale)
                    q = torch.clamp(torch.round(w_flat / scale[:, None] + zero_point[:, None]), qmin, qmax_w)
                    module.weight.data = ((q - zero_point[:, None]) * scale[:, None]).view_as(w)
    else:  # per_tensor
        for _, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                w = module.weight.data
                w_np = w.cpu().numpy().ravel()
                if symmetric:
                    scale = compute_scale_symmetric(w_np, qmax_w)
                    zero_point = 0
                else:
                    scale, zero_point = compute_scale_asymmetric(w_np, qmin, qmax_w)
                module.weight.data = quantize_tensor(w, scale, zero_point, qmin, qmax_w)
    return model


def export_quantized_weights_npz(model: torch.nn.Module, path: str) -> None:
    """
    Save quantized model weights (int) to a .npz file for inspection or hardware use.
    Accepts a model (assumed to contain quantized tensors) and writes integer reprs.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    state = model.state_dict()
    weights = {}
    for name, tensor in state.items():
        if isinstance(tensor, torch.Tensor) and tensor.dtype in (torch.qint8, torch.quint8, torch.int8, torch.uint8):
            weights[name] = tensor.int_repr().cpu().numpy()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **weights)
    print(f"[OK] Quantized weights saved to {path}")

def export_quant_from_pt(
    weights_path: str,
    out_path: str,
    bits: int = 5,
    per_channel: bool = True,
    symmetric: bool = True,
) -> List[str]:
    """
    Quantize raw float weights from a PyTorch checkpoint to integer codes + scales
    and store them in an NPZ archive (variable shapes supported via object arrays).
    """
    obj = torch.load(weights_path, map_location="cpu")
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1
    qmin = -qmax if symmetric else 0

    names: List[str] = []
    q_list: List[np.ndarray] = []
    scale_list: List[np.ndarray] = []

    for name, w in state.items():
        if not isinstance(w, torch.Tensor):
            continue
        if "weight" not in name:
            continue
        if w.ndim < 2:  # skip biases / norms
            continue
        wf = w.float().cpu()
        if per_channel:
            oc = wf.shape[0]
            flat = wf.view(oc, -1)
            if symmetric:
                max_abs = flat.abs().max(dim=1).values
            else:
                max_abs = flat.max(dim=1).values
            scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)
            q = torch.round(flat / scale[:, None]).clamp(qmin, qmax).to(torch.int16)
            names.append(name)
            q_list.append(q.numpy())
            scale_list.append(scale.numpy())
        else:
            max_abs = wf.abs().max() if symmetric else wf.max()
            if max_abs == 0:
                continue
            scale = (max_abs / qmax).item()
            q = torch.round(wf / scale).clamp(qmin, qmax).to(torch.int16)
            names.append(name)
            q_list.append(q.numpy())
            scale_list.append(np.array(scale, dtype=np.float32))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        names=np.array(names, dtype=object),
        q_list=np.array(q_list, dtype=object),
        scale_list=np.array(scale_list, dtype=object),
        bits=np.array(bits),
        symmetric=np.array(symmetric),
        per_channel=np.array(per_channel),
    )
    print(f"[OK] Quantized {len(names)} tensors -> {out_path}")
    return names


def quantize_model_ptq(
    model: torch.nn.Module,
    weight_bits: int = 8,
    act_bits: int = 8,
    scheme: str = "global",
    symmetric: bool = True,
    global_percentile: float = 100.0
) -> torch.nn.Module:
    """
    Post-training quantization (PTQ) for both weights and activations.
    
    This function quantizes a trained model without any retraining. It:
    1. Quantizes model weights (Conv/Linear layers) to lower precision
    2. Adds input quantization to simulate low-precision inference
    
    Args:
        model: PyTorch model to quantize
        weight_bits: Bit-width for weights (use 32 or 'float' to skip)
        act_bits: Bit-width for activations (use 32 or 'float' to skip)
        scheme: Quantization scheme ('per_tensor', 'per_channel', or 'global')
        symmetric: Use symmetric quantization (range: -qmax to qmax)
        global_percentile: Percentile for global scale computation (100 = no clipping)
    
    Returns:
        Quantized model copy (original model unchanged)
    """
    
    # Helper: Check if bit-width specification means "keep as float"
    # Accepts: "float", "fp", "fp32", "32", or numeric >= 16
    def is_float_spec(b) -> bool:
        return str(b).lower() in ("float", "fp", "fp32", "32")
    
    # Helper: Determine if we should skip quantization entirely
    # Skip if both weights and activations are high-precision (>=16 bit)
    def treat_as_float(w_bits, a_bits) -> bool:
        return (is_float_spec(w_bits) and is_float_spec(a_bits)) or (
            (not is_float_spec(w_bits) and int(w_bits) >= 16) and
            (not is_float_spec(a_bits) and int(a_bits) >= 16)
        )
    
    # Fast path: If both weights and activations are high precision, just return a copy
    # This avoids unnecessary processing for baseline (float) evaluations
    if treat_as_float(weight_bits, act_bits):
        return deepcopy(model)
    
    # Create a copy on CPU in eval mode (PTQ doesn't require gradients)
    model = deepcopy(model).cpu().eval()
    
    # ========== STEP 1: Quantize Weights ==========
    # Quantize all Conv1d/Linear layer weights in-place
    # This simulates storing weights in lower precision (e.g., 4-bit, 8-bit)
    if not is_float_spec(weight_bits):
        w_bits = int(weight_bits)
        model = quantize_model_weights(
            model,
            bits=w_bits,
            scheme=scheme,              # per_tensor, per_channel, or global
            symmetric=symmetric,         # symmetric: [-qmax, qmax], asymmetric: [0, qmax]
            global_percentile=global_percentile  # Clip outliers if < 100
        )
    
    # ========== STEP 2: Quantize Activations ==========
    # If activation quantization is disabled (float or >=16 bit), return model with quantized weights only
    if is_float_spec(act_bits) or int(act_bits) >= 16:
        return model
    
    # Compute quantization range for activations (symmetric quantization)
    # Example: 8-bit → qmin=-128, qmax=127
    act_bits_i = int(act_bits)
    qmin = -(2 ** (act_bits_i - 1))
    qmax = (2 ** (act_bits_i - 1)) - 1
    
    # Save original forward method
    orig_forward = model.forward
    
    # Wrap the forward pass to quantize inputs (activations)
    # This simulates running inference with low-precision activations
    @torch.no_grad()
    def quantized_forward(x: torch.Tensor) -> torch.Tensor:
        # Find the maximum absolute value in the input tensor
        max_abs = x.abs().max()
        
        # Skip quantization if input is all zeros (avoid division by zero)
        if max_abs == 0:
            return orig_forward(x)
        
        # Compute quantization scale: scale = max_val / qmax
        # This ensures the full dynamic range maps to [qmin, qmax]
        scale = max_abs / qmax
        
        # Quantize: x_float → x_int (round and clamp to valid range)
        x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
        
        # Dequantize: x_int → x_float (convert back to floating point)
        # This simulates the quantization error introduced by low-precision storage
        x_dq = x_q * scale
        
        # Run the model with quantized input
        return orig_forward(x_dq)
    
    # Replace the model's forward method with the quantized version
    model.forward = quantized_forward
    return model