import os
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader
from typing import List, Tuple, Dict, Any

from train.utils import load_state_dict_forgiving, evaluate_model
from config import load_config
from model.model import DilatedTCN
from data_loader.utils import make_datasets, get_num_classes
from quantization.core import quantize_model_ptq


def is_float_spec(b: Any) -> bool:
    """Check if bit-width specification indicates floating point."""
    return str(b).lower() in ("float", "fp", "fp32", "32")


def treat_as_float(w_bits: Any, a_bits: Any) -> bool:
    """Check if weight/activation bits should bypass quantization."""
    return (is_float_spec(w_bits) and is_float_spec(a_bits)) or (
        (not is_float_spec(w_bits) and int(w_bits) >= 16) and
        (not is_float_spec(a_bits) and int(a_bits) >= 16)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", choices=["val", "train", "test", "all", "full"], default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--bits", type=str, default="float,8,4,2")
    parser.add_argument("--act-bits", type=str, default=None)
    parser.add_argument("--scheme", choices=["per_tensor", "global", "per_channel"], default="global")
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--global-percentile", type=float, default=100.0,
                        help="Percentile for global quantization scale (default: 100, i.e. max).")
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)
    device: torch.device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    # Get all splits, then select/compose requested eval loader
    train_loader, val_loader, test_loader = make_datasets(cfg, which="all", batch_size=args.batch_size)
    if args.dataset == "train":
        dl = train_loader
    elif args.dataset == "val":
        dl = val_loader
    elif args.dataset == "test":
        dl = test_loader
    else:
        combo = ConcatDataset([train_loader.dataset, val_loader.dataset, test_loader.dataset])
        nw = int(cfg.get("train", {}).get("num_workers", 0))
        pin = bool(cfg.get("train", {}).get("pin_memory", False))
        dl = DataLoader(combo, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    num_classes: int = get_num_classes(cfg)
    task_type: str = cfg["task"]["type"]

    # Build model using the model class factory
    model: torch.nn.Module = DilatedTCN.from_config(cfg)
    # Load weights using the shared forgiving loader
    model = load_state_dict_forgiving(model, args.weights, device=torch.device("cpu"))
    model = model.to(device).eval()

    per_channel: bool = args.scheme == "per_channel"
    symmetric: bool = args.symmetric

    def parse_list(s: str) -> List[str]:
        return [t.strip() for t in s.split(",") if t.strip()]

    bits_list: List[str] = parse_list(args.bits)
    act_bits_list: List[str] = bits_list if args.act_bits is None else parse_list(args.act_bits)

    results: List[Tuple[str, float, float, float, float]] = []
    for i, wb in enumerate(bits_list):
        ab = act_bits_list[i] if i < len(act_bits_list) else act_bits_list[-1]

        if treat_as_float(wb, ab):
            tag = "float"
            from copy import deepcopy
            model_q = deepcopy(model)
        else:
            w_bits_i = 32 if is_float_spec(wb) else int(wb)
            a_bits_i = 32 if is_float_spec(ab) else int(ab)
            tag = "float" if (w_bits_i >= 16 and a_bits_i >= 16) else f"{w_bits_i}w{a_bits_i}a"
            model_q = quantize_model_ptq(
                model, w_bits_i, a_bits_i, scheme=args.scheme, symmetric=symmetric,
                global_percentile=args.global_percentile
            )

        acc, prec, rec, f1 = evaluate_model(model_q.to(device), dl, device, task_type, num_classes=num_classes)
        results.append((tag, acc, prec, rec, f1))
        print(f"[{tag:>8}] Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

    # --- plotting unchanged (use results list) ---
    plots_dir: str = os.path.join(cfg["output"]["plots_dir"], "eval")
    os.makedirs(plots_dir, exist_ok=True)
    out_path: str = os.path.join(plots_dir, "ptq_metrics_vs_bits.png")

    labels: List[str] = []
    accs: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    f1s: List[float] = []
    for tag, acc, prec, rec, f1 in results:
        labels.append(tag)
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

    plt.figure(figsize=(9, 5))
    for vals, lbl, mk in zip([accs, precs, recs, f1s],
                              ["Accuracy", "Precision", "Recall", "F1"],
                              ["o", "s", "^", "d"]):
        plt.plot(range(len(labels)), vals, marker=mk, label=lbl)

    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.ylabel("Metric")
    plt.xlabel("Quant setting")
    plt.title("PTQ Metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved plot: {out_path}")


if __name__ == "__main__":
    main()
