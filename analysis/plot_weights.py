"""
Visualize trained model weights (state_dict):

1) Overall histogram (float, weights+biases) with zero/non-zero counts.
2) Per-layer histograms (weights & biases) with zero/non-zero counts.
3) Multi-quant figure (weights only):
   - 'float' panel: value-domain histogram (continuous).
   - For quantized variants (e.g., 8/4/3 bit): histogram of quantized INTEGER CODES,
     so bins align exactly with quantization levels (bin width = 1, centered on each level).
   - Each panel shows total zeros and non-zeros. For quantized panels, zeros are exact code=0.

Usage examples:
  python -m analysis.plot_weights --weights model_weights.pt
  python -m analysis.plot_weights --weights model_weights.pt --zero-threshold 1e-6
  python -m analysis.plot_weights --weights model_weights.pt --quant-bits 8,4,3 --quant-scheme global --global-percentile 100
"""

import os
import argparse
from typing import Dict, Optional, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from train.utils import build_model_from_cfg, load_config, deep_update
from quantization.core import (
    quantize_state_dict_to_codes,
    qmax_for_bits
)


def fold_batchnorm_into_weights(state_dict):
    """
    Returns a new state_dict with BatchNorm1d folded into preceding Conv1d/Linear layers.
    Only supports standard PyTorch naming conventions: <layer>.weight, <layer>.bias, <bn>.weight, <bn>.bias, <bn>.running_mean, <bn>.running_var, <bn>.eps
    """
    import copy
    sd = copy.deepcopy(state_dict)
    folded = {}
    keys = list(sd.keys())
    for k in keys:
        if not (k.endswith('.weight') and ('.bn' in k or '.batchnorm' in k or '.norm' in k or 'bn' in k.split('.')[-2])):
            continue
        # Try to find preceding conv/linear layer
        parts = k.split('.')
        if len(parts) < 2:
            continue
        prefix = '.'.join(parts[:-1])
        # Try to find previous layer (same prefix minus last part)
        parent = '.'.join(parts[:-2])
        # Try common patterns: block.bn, block.conv, block.linear
        for cand in ['conv1', 'conv2', 'conv', 'linear', 'fc']:
            w_key = parent + '.' + cand + '.weight' if parent else cand + '.weight'
            b_key = parent + '.' + cand + '.bias' if parent else cand + '.bias'
            if w_key in sd:
                # Fold BatchNorm into this layer
                W = sd[w_key].clone()
                b = sd[b_key].clone() if b_key in sd else torch.zeros(W.shape[0])
                gamma = sd[k]
                beta = sd[prefix + '.bias'] if prefix + '.bias' in sd else torch.zeros_like(gamma)
                mean = sd[prefix + '.running_mean']
                var = sd[prefix + '.running_var']
                eps = sd[prefix + '.eps'] if prefix + '.eps' in sd else 1e-5
                std = (var + eps).sqrt()
                W_fold = W * (gamma / std).reshape([-1] + [1]*(W.dim()-1))
                b_fold = (b - mean) / std * gamma + beta
                folded[w_key] = W_fold
                folded[b_key] = b_fold
    # Update state_dict with folded weights
    for k, v in folded.items():
        sd[k] = v
    return sd

# -----------------------------
# IO helpers
# -----------------------------
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


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Zero/non-zero counting
# -----------------------------
def zero_nonzero_counts_from_array(arr: np.ndarray, zero_threshold: float = 0.0) -> Tuple[int, int]:
    if zero_threshold <= 0.0:
        zeros = int(np.count_nonzero(arr == 0.0))
    else:
        zeros = int(np.count_nonzero(np.abs(arr) <= zero_threshold))
    nonzeros = int(arr.size - zeros)
    return zeros, nonzeros


def zero_nonzero_counts(tensor: torch.Tensor, zero_threshold: float = 0.0) -> Tuple[int, int]:
    arr = tensor.detach().cpu().numpy().ravel()
    return zero_nonzero_counts_from_array(arr, zero_threshold)


# -----------------------------
# Plot helpers
# -----------------------------
def hist_with_counts(
    tensor: torch.Tensor,
    title: str,
    out_path: str,
    bins: int = 100,
    zero_threshold: float = 0.0,
    show: bool = False,
):
    """
    Save a histogram for the given tensor and annotate with zero/non-zero counts.
    """
    ensure_dir(os.path.dirname(out_path))
    arr = tensor.detach().cpu().numpy().ravel()
    zeros, nonzeros = zero_nonzero_counts_from_array(arr, zero_threshold)

    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.3)

    txt = (
        f"total: {arr.size}\n"
        f"zero{' (|w|<=' + str(zero_threshold) + ')' if zero_threshold>0 else ''}: {zeros}\n"
        f"non-zero: {nonzeros}"
    )
    plt.gca().text(
        0.98, 0.95, txt,
        transform=plt.gca().transAxes,
        ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

    plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[OK] Saved {out_path}")


def overall_histogram(state_dict: Dict[str, torch.Tensor], outdir: str, bins: int, zero_threshold: float, show: bool, suffix: str = ""):
    """Plot a single histogram of all floating-point .weight tensors (no biases), with counts."""
    ensure_dir(outdir)
    values = []
    for k, v in state_dict.items():
        if not k.endswith('.weight'):
            continue
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            values.append(v.detach().cpu().numpy().ravel())
    if not values:
        print("[WARN] No floating-point .weight tensors found in state_dict.")
        return

    arr = np.concatenate(values, axis=0)
    zeros, nonzeros = zero_nonzero_counts_from_array(arr, zero_threshold)

    plt.figure(figsize=(9, 5))
    plt.hist(arr, bins=bins)
    plt.title(f"All weights histogram (float){suffix}")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.3)

    txt = (
        f"total: {arr.size}\n"
        f"zero{' (|w|<=' + str(zero_threshold) + ')' if zero_threshold>0 else ''}: {zeros}\n"
        f"non-zero: {nonzeros}"
    )
    plt.gca().text(
        0.98, 0.95, txt,
        transform=plt.gca().transAxes,
        ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

    out_path = os.path.join(outdir, f"all_weights_hist{suffix}.png")
    plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[OK] Saved {out_path}")


# -----------------------------
# Collect weight arrays
# -----------------------------
def collect_weight_arrays(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    """Collect flattened arrays for all .weight tensors (float)."""
    arrs = []
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor):
            continue
        if not t.dtype.is_floating_point:
            continue
        if name.endswith(".weight_fake_quant.scale"):
            continue  # skip fake quant metadata
        if name.endswith(".weight"):
            arrs.append(t.detach().cpu().numpy().ravel())
    return arrs



# -----------------------------
# Multi-quant figure (integer-code hist for quantized variants)
# -----------------------------
def plot_multi_quant_histograms(
    state_dict: Dict[str, torch.Tensor],
    outdir: str,
    bits_list: List[int],
    scheme: str,
    bins_float: int,
    zero_threshold: float,
    show: bool,
    global_percentile: float = 100.0,
):
    """
    Create a figure with subplots:
      - Float baseline: value-domain histogram (weights only, continuous).
      - Each quantized variant: integer-code histogram with bins at each quant level.
    Saves to weights_hist_multi_quant.png
    """
    ensure_dir(outdir)

    # Base float weights (weights only, no biases)
    float_arrs = collect_weight_arrays(state_dict)
    if not float_arrs:
        print("[WARN] No .weight tensors found; skipping multi-quant histogram.")
        return
    float_arr = np.concatenate(float_arrs, axis=0)
    f_zeros, f_nonzeros = zero_nonzero_counts_from_array(float_arr, zero_threshold)
    variants = [("float (value)", float_arr, f_zeros, f_nonzeros, None)]  # last field: bins (None for float)

    # Quantized variants (integer codes)
    for b in bits_list:
        qmax = qmax_for_bits(b)
        q_codes = quantize_state_dict_to_codes(state_dict, bits=b, scheme=scheme, global_percentile=global_percentile)
        edges = np.arange(-qmax - 0.5, qmax + 1.5, 1.0)
        q_zeros = int(np.count_nonzero(q_codes == 0))
        q_nonzeros = int(q_codes.size - q_zeros)
        variants.append((f"{b}-bit ({scheme})", q_codes, q_zeros, q_nonzeros, edges))

    # Layout
    n = len(variants)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, data, zeros, nonzeros, edges) in zip(axes, variants):
        if edges is None:
            # float value-domain histogram
            ax.hist(data, bins=bins_float)
            ax.set_xlabel("value")
        else:
            # integer-code histogram with aligned bins
            ax.hist(data, bins=edges)
            ax.set_xlabel("quantized level (integer code)")
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)

        txt = (
            f"total: {data.size}\n"
            f"zero: {zeros}\n"
            f"non-zero: {nonzeros}"
        )
        ax.text(
            0.98, 0.95, txt,
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )

    # Hide extra axes
    for ax in axes[len(variants):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = os.path.join(outdir, "weights_hist_multi_quant.png")
    plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_actual_quantized_weights(state_dict, outdir, show=False):
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(outdir, exist_ok=True)
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.dtype in (
            torch.qint8, torch.quint8, torch.int8, torch.uint8
        ):
            arr = tensor.int_repr().cpu().numpy().ravel()
            plt.figure(figsize=(8, 5))
            plt.hist(arr, bins=np.arange(arr.min()-0.5, arr.max()+1.5, 1), color='C0')
            plt.title(f"{name} (actual quantized codes)")
            plt.xlabel("quantized integer code")
            plt.ylabel("count")
            plt.grid(True, axis="y", alpha=0.3)
            out_path = os.path.join(outdir, f"{name.replace('.', '_')}_actual_quant_hist.png")
            plt.savefig(out_path, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()
            print(f"[OK] Saved {out_path}")
            
# -----------------------------
# Per-parameter figures (weights & biases)
# -----------------------------
def plot_per_layer_histograms(
    state_dict: Dict[str, torch.Tensor],
    outdir: str,
    bins: int,
    zero_threshold: float,
    show: bool,
):
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not tensor.dtype.is_floating_point:
            continue

        if name.endswith(".weight"):
            fname = f"{name.replace('.', '_')}_hist.png"
            out_path = os.path.join(outdir, fname)
            title = f"[{name}] weight histogram"
            hist_with_counts(tensor, title=title, out_path=out_path, bins=bins,
                             zero_threshold=zero_threshold, show=show)

            # Matching bias (if present)
            bias_name = name[:-len(".weight")] + ".bias"
            bias_tensor: Optional[torch.Tensor] = state_dict.get(bias_name, None)
            if isinstance(bias_tensor, torch.Tensor) and bias_tensor.dtype.is_floating_point:
                bias_fname = f"{bias_name.replace('.', '_')}_hist.png"
                bias_out = os.path.join(outdir, bias_fname)
                bias_title = f"[{bias_name}] bias histogram"
                hist_with_counts(bias_tensor, title=bias_title, out_path=bias_out, bins=bins,
                                 zero_threshold=zero_threshold, show=show)


# -----------------------------
# Main
# -----------------------------
def parse_bits_list(bits_str: str) -> List[int]:
    bits = []
    for s in bits_str.split(","):
        s = s.strip().lower()
        if not s:
            continue
        if s in ("f", "fp", "float", "fp32", "32"):
            # Float baseline is implicit in multi-quant figure
            continue
        b = int(s)
        if b < 2 or b > 16:
            raise ValueError("Bitwidths must be integers in [2..16] (float baseline is implicit).")
        bits.append(b)
    return bits


def main():
    # --- Print all bias parameters with summary ---
    
    parser = argparse.ArgumentParser(description="Visualize model weights with histograms and multi-quant (aligned bins).")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt (state_dict or checkpoint).")
    parser.add_argument("--outdir", type=str, default="plots/weights_viz", help="Directory to save figures.")
    parser.add_argument("--bins", type=int, default=100, help="Bins for float histograms (overall & per-parameter).")
    parser.add_argument("--zero-threshold", type=float, default=0.0,
                        help="Treat |w| <= threshold as zero for float hist counts (not used for integer-code hist).")
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    parser.add_argument("--quant-bits", type=str, default="8,4,3",
                        help="Comma-separated bit widths for multi-quant figure (float is implicit).")
    parser.add_argument("--quant-scheme", type=str, choices=["per_channel", "per_tensor", "global"], default="per_tensor",
                        help="Weight quantization scheme for the multi-quant figure.")
    parser.add_argument("--global-percentile", type=float, default=100.0,
                        help="Percentile for global quantization scale (default: 100, i.e. max).")
    parser.add_argument("--per-layer", action="store_true",
                        help="If set, save per-parameter (layer) histograms. Default: off.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    sd = load_state_dict(args.weights)
    print("[DEBUG] All state_dict keys:", list(sd.keys()))

    # --- Print loaded state_dict keys and tensor shapes ---
    print("\n[STATE_DICT CONTENTS]")
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:<40} {tuple(v.shape)}")

    print("\n[BIAS PARAMETERS]")
    for k, v in sd.items():
        if k.endswith('.bias') and isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy().ravel()
            print(f"{k:<40} shape={arr.shape} min={arr.min():.4g} max={arr.max():.4g} mean={arr.mean():.4g} sample={arr[:5]}")

    # --- Print summary of all tensors ---
    print("\n[CHECKPOINT TENSOR SUMMARY]")
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy().ravel()
            dtype = v.dtype
            print(f"{k:<40} shape={v.shape} dtype={dtype} min={arr.min():.4g} max={arr.max():.4g} mean={arr.mean():.4g} sample={arr[:5]}")
        else:
            print(f"{k:<40} type={type(v)}")

    # --- Detect quantized weights ---
    has_quantized = any(
        isinstance(t, torch.Tensor) and t.dtype in (torch.qint8, torch.quint8, torch.int8, torch.uint8)
        for t in sd.values()
    )


    if has_quantized:
        print("[INFO] Detected quantized weights. Plotting actual quantized histograms only.")
        plot_actual_quantized_weights(sd, args.outdir, show=args.show)
    else:
        print("[INFO] Detected floating-point weights. Plotting FP and simulated quantization histograms.")
        # Print actual dtype of first floating-point tensor
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                print(f"[INFO] Floating-point dtype: {v.dtype}")
                break

        # --- Print min/max of all_weights (unfolded) ---
        all_weights = []
        for k, v in sd.items():
            if k.endswith('.weight') and isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                all_weights.append(v.detach().cpu().numpy().ravel())
        if all_weights:
            all_weights_arr = np.concatenate(all_weights, axis=0)
            print(f"[ALL WEIGHTS] min={all_weights_arr.min():.6g} max={all_weights_arr.max():.6g}")
        else:
            all_weights_arr = None

        overall_histogram(sd, args.outdir, bins=args.bins, zero_threshold=args.zero_threshold, show=args.show)

        # --- Print min/max of all_weights_folded ---
        sd_folded = fold_batchnorm_into_weights(sd)
        all_weights_folded = []
        for k, v in sd_folded.items():
            if k.endswith('.weight') and isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                all_weights_folded.append(v.detach().cpu().numpy().ravel())
        if all_weights_folded:
            all_weights_folded_arr = np.concatenate(all_weights_folded, axis=0)
            print(f"[ALL WEIGHTS FOLDED] min={all_weights_folded_arr.min():.6g} max={all_weights_folded_arr.max():.6g}")
        else:
            all_weights_folded_arr = None

        overall_histogram(sd_folded, args.outdir, bins=args.bins, zero_threshold=args.zero_threshold, show=args.show, suffix="_folded")

        if args.per_layer:
            plot_per_layer_histograms(sd, args.outdir, bins=args.bins, zero_threshold=args.zero_threshold, show=args.show)

        bits_list = parse_bits_list(args.quant_bits)
        if bits_list:
            for b in bits_list:
                q_codes = quantize_state_dict_to_codes(sd, bits=b, scheme=args.quant_scheme, global_percentile=args.global_percentile)
                if q_codes.size > 0:
                    print(f"[QUANTIZED {b}-bit {args.quant_scheme}] min={q_codes.min()} max={q_codes.max()}")
                else:
                    print(f"[QUANTIZED {b}-bit {args.quant_scheme}] No quantized codes found.")
            plot_multi_quant_histograms(
                state_dict=sd,
                outdir=args.outdir,
                bits_list=bits_list,
                scheme=args.quant_scheme,
                bins_float=args.bins,
                zero_threshold=args.zero_threshold,
                show=args.show,
                global_percentile=args.global_percentile,
            )
        else:
            print("[INFO] No quant bits requested; skipped multi-quant figure.")

    print(f"[DONE] Outputs written to: {args.outdir}")

if __name__ == "__main__":
    main()
