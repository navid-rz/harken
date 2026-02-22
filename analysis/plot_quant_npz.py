import argparse, os, numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def try_extract_items(npz):
    # Supported layouts:
    # 1) keys like "tcn_layers.0.weight.q" and "...scale"
    # 2) arrays: names (list of str), q_list (object array), scale_list (object array)
    # 3) QAT format: actual weight names with quantized float values
    if "names" in npz and "q_list" in npz:
        names = [str(n) for n in npz["names"].tolist()]
        q_list = [np.array(a) for a in npz["q_list"]]
        scales = [None]*len(q_list)
        if "scale_list" in npz:
            scales = [np.array(a) for a in npz["scale_list"]]
        return list(zip(names, q_list, scales))

    # Check for QAT format (quantized float weights with layer names)
    qat_items = []
    for k in npz.files:
        if 'weight' in k and not k.endswith('.q') and not k.endswith('.scale'):
            weight_data = np.array(npz[k])
            # Always include weight tensors from .npz files (they're intended to be quantized)
            qat_items.append((k, weight_data, None))
    
    if qat_items:
        print(f"[INFO] Detected QAT format: {len(qat_items)} quantized weight tensors")
        return qat_items

    # Fallback: group by basename before suffix ".q" / ".scale"
    groups = {}
    for k in npz.files:
        if k.endswith(".q"):
            base = k[:-2]
            groups.setdefault(base, {})["q"] = np.array(npz[k])
        elif k.endswith(".scale") or k.endswith(".scales"):
            base = k[: -len(".scale")] if k.endswith(".scale") else k[: -len(".scales")]
            groups.setdefault(base, {})["scale"] = np.array(npz[k])
    items = []
    for base, d in groups.items():
        if "q" in d:
            items.append((base, d["q"], d.get("scale", None)))
    return items

def plot_hist_weights(name, q, weight_hist_dir, qmin=None, qmax=None, annotate_counts=False, is_float_weights=False):
    q = np.asarray(q)
    flat = q.reshape(-1)
    
    if is_float_weights:
        # For quantized float weights, show actual weight values
        unique_vals = np.unique(flat)
        
        # Create unique values plot if reasonable number of unique values
        if len(unique_vals) <= 50:  
            plt.figure(figsize=(10,6))
            counts = [np.sum(flat == val) for val in unique_vals]
            plt.bar(range(len(unique_vals)), counts, alpha=0.8)
            unique_title = f"Full Network Unique Quantized Values" if name == "FULL_NETWORK" else f"Unique quantized values: {name}"
            plt.title(unique_title)
            plt.xlabel("Quantization Level")
            plt.ylabel("Count")
            plt.xticks(range(len(unique_vals)), [f"{val:.3f}" for val in unique_vals], rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Annotate with stats
            total = flat.size
            zeros = int((flat == 0).sum())
            nonzeros = total - zeros
            weight_min, weight_max = flat.min(), flat.max()
            txt = f"total = {total:,}\nnon-zero = {nonzeros:,}\nzeros = {zeros:,}\nunique = {len(unique_vals)}\nrange = [{weight_min:.6f}, {weight_max:.6f}]"
            ax = plt.gca()
            ax.text(
                0.98, 0.95, txt,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
            )
            plt.tight_layout()
            
            unique_path = os.path.join(weight_hist_dir, f"{name.replace('.', '_').replace('/', '_')}_unique_vals.png")
            plt.savefig(unique_path)
            plt.close()
            return unique_path
            
        else:
            # Fall back to histogram for too many unique values
            plt.figure(figsize=(10,6))
            plt.hist(flat, bins=50, edgecolor="k", alpha=0.8)
            title = f"Full Network Quantized Weights" if name == "FULL_NETWORK" else f"Quantized weight distribution: {name}"
            plt.title(title)
            plt.xlabel("Weight value")
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            
            # Annotate unique values and range
            total = flat.size
            zeros = int((flat == 0).sum())
            nonzeros = total - zeros
            weight_min, weight_max = flat.min(), flat.max()
            txt = f"total = {total:,}\nnon-zero = {nonzeros:,}\nzeros = {zeros:,}\nunique = {len(unique_vals)}\nrange = [{weight_min:.6f}, {weight_max:.6f}]"
            ax = plt.gca()
            ax.text(
                0.98, 0.95, txt,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
            )
            
    else:
        # Original integer code plotting
        low = flat.min() if qmin is None else qmin
        high = flat.max() if qmax is None else qmax
        # Bin centers at every integer from low..high
        bins = np.arange(low - 0.5, high + 1.5, 1.0)  # width = 1
        plt.figure(figsize=(7,4))
        plt.hist(flat, bins=bins, edgecolor="k", alpha=0.8)
        title = f"Full Network Integer Codes" if name == "FULL_NETWORK" else f"Integer code usage: {name}"
        plt.title(title)
        plt.xlabel("q (integer code)")
        plt.ylabel("count")
        plt.grid(True, alpha=0.3)
        
        # Annotate counts for ALL_PARAMS or when requested
        if annotate_counts or name in ["ALL_PARAMS", "FULL_NETWORK"]:
            total = flat.size
            zeros = int((flat == 0).sum())
            nonzeros = total - zeros
            txt = f"total = {total:,}\nnon-zero = {nonzeros:,}\nzeros = {zeros:,}"
            ax = plt.gca()
            ax.text(
                0.98, 0.95, txt,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
            )
    
    # Only reach here for non-float weights (integers) or fallback histogram
    ensure_dir(weight_hist_dir)
    suffix = "_weight_hist" if is_float_weights else "_q_hist" 
    path = os.path.join(weight_hist_dir, f"{name.replace('.', '_').replace('/', '_')}{suffix}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def main():
    ap = argparse.ArgumentParser(description="Plot histograms of quantized weights or integer codes from NPZ export.")
    ap.add_argument("--npz", required=True, help="Path to exported quant .npz")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--bits", type=int, default=None, help="Optional: number of bits (only used to set q range for integer codes)")
    args = ap.parse_args()

    # Create weight_histograms subdirectory
    weight_hist_dir = os.path.join(args.outdir, "weight_histograms")
    ensure_dir(weight_hist_dir)
    npz = np.load(args.npz, allow_pickle=True)
    items = try_extract_items(npz)
    if not items:
        print("[ERROR] Could not find any quantized arrays in NPZ. Keys:", list(npz.files))
        return

    # Detect if we're dealing with float weights or integer codes
    sample_data = items[0][1] if len(items) > 0 else None
    is_float_weights = False
    if sample_data is not None:
        is_float_weights = sample_data.dtype == np.float32 or sample_data.dtype == np.float64
        data_type = "quantized float weights" if is_float_weights else "integer codes"
        print(f"[INFO] Detected data type: {data_type}")

    # Optional fixed q range from bits (only for integer codes)
    qmin = qmax = None
    if args.bits and not is_float_weights:
        qmax = (1 << (args.bits - 1)) - 1
        qmin = -qmax

    all_q = []
    layer_count = 0
    for name, q, scale in items:
        if q is None:
            continue
        all_q.append(q.reshape(-1))
        uniq = np.unique(q).size
        data_range = f"[{q.min():.6f}, {q.max():.6f}]" if is_float_weights else f"[{q.min()}, {q.max()}]"
        print(f"[INFO] {name}: unique values = {uniq} | range = {data_range}")
        layer_count += 1

    # Combined histogram only
    if all_q:
        flat_all = np.concatenate(all_q, axis=0)
        out = plot_hist_weights("FULL_NETWORK", flat_all, weight_hist_dir, qmin=qmin, qmax=qmax, annotate_counts=True, is_float_weights=is_float_weights)
        uniq = np.unique(flat_all).size
        data_range = f"[{flat_all.min():.6f}, {flat_all.max():.6f}]" if is_float_weights else f"[{flat_all.min()}, {flat_all.max()}]"
        print(f"[OK] Full network plot saved: {out}")
        print(f"     Layers processed: {layer_count} | Total unique values: {uniq} | Range: {data_range}")

if __name__ == "__main__":
    main()