import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, Subset
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from copy import deepcopy
import tempfile

from config import load_config
from inference.model import KeywordModel
from data_loader.utils import make_datasets
from quantization.core import quantize_model_weights, quantize_to_log2


def apply_log2_quantization(model: nn.Module, num_levels: int, global_percentile: float = 100.0, 
                           normalize_to_unit: bool = False) -> nn.Module:
    """
    Apply log2 quantization to model weights using global scaling.
    
    Args:
        model: PyTorch model to quantize  
        num_levels: Number of log2 levels (creates range ±2^num_levels)
        global_percentile: Percentile for global max computation
        normalize_to_unit: If True, normalize to [-1,+1] for hardware deployment
        
    Returns:
        New model with log2 quantized weights
    """
    model_q = deepcopy(model)
    
    # Collect all weight parameters for global percentile computation
    all_weights = []
    for module in model_q.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            all_weights.append(module.weight.data.cpu().numpy().flatten())
    
    if not all_weights:
        return model_q
        
    # Compute global percentile across all weights
    all_weights_flat = np.concatenate(all_weights)
    global_max = np.percentile(np.abs(all_weights_flat), global_percentile)
    
    print(f"[LOG2 QUANTIZATION DEBUG] normalize_to_unit={normalize_to_unit}")
    print(f"  Original weights: min={all_weights_flat.min():.6f}, max={all_weights_flat.max():.6f}")
    print(f"  Global max (percentile): {global_max:.6f}")
    
    # Apply log2 quantization to each layer using global max
    quantized_weights = []
    for module in model_q.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            w_np = module.weight.data.cpu().numpy()
            w_quantized = quantize_to_log2(w_np, num_levels, global_max, normalize_to_unit)
            module.weight.data = torch.from_numpy(w_quantized).float().to(module.weight.device)
            quantized_weights.append(w_quantized.flatten())
    
    if quantized_weights:
        all_quantized = np.concatenate(quantized_weights)
        print(f"  Quantized weights: min={all_quantized.min():.6f}, max={all_quantized.max():.6f}")
        unique_values = np.unique(all_quantized)
        print(f"  Unique quantized values: {len(unique_values)} values")
        print(f"  Sample values: {unique_values[:10]}")
            
    return model_q


def apply_linear_quantization(model: nn.Module, weight_bits: int, global_percentile: float = 100.0,
                             normalize_to_unit: bool = False) -> nn.Module:
    """
    Apply linear quantization to model weights using global scaling.
    
    Args:
        model: PyTorch model to quantize
        weight_bits: Number of bits for linear quantization
        global_percentile: Percentile for global max computation
        normalize_to_unit: If True, normalize to [-1,+1] for hardware deployment
        
    Returns:
        New model with linear quantized weights
    """
    model_q = deepcopy(model)
    
    # Collect original weight stats
    all_weights = []
    for module in model_q.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            all_weights.append(module.weight.data.cpu().numpy().flatten())
    
    if all_weights:
        all_weights_orig = np.concatenate(all_weights)
        print(f"[LINEAR QUANTIZATION DEBUG] normalize_to_unit={normalize_to_unit}")
        print(f"  Original weights: min={all_weights_orig.min():.6f}, max={all_weights_orig.max():.6f}")
    
    # Use the existing quantization framework - it modifies the model in place
    quantized_model = quantize_model_weights(
        model_q, 
        bits=weight_bits,
        scheme="global",
        symmetric=True,
        global_percentile=global_percentile,
        normalize_to_unit=normalize_to_unit
    )
    
    # Check quantized weight stats
    quantized_weights = []
    for module in quantized_model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            quantized_weights.append(module.weight.data.cpu().numpy().flatten())
    
    if quantized_weights:
        all_quantized = np.concatenate(quantized_weights)
        print(f"  Quantized weights: min={all_quantized.min():.6f}, max={all_quantized.max():.6f}")
        unique_values = np.unique(all_quantized)
        print(f"  Unique quantized values: {len(unique_values)} values")
        print(f"  Sample values: {unique_values[:10]}")
    
    return quantized_model


def create_subset_loader(dataloader: DataLoader, subset_size: int) -> DataLoader:
    """
    Create a subset DataLoader for faster testing.
    
    Args:
        dataloader: Original DataLoader
        subset_size: Number of samples to include in subset
        
    Returns:
        New DataLoader with subset of original data
    """
    if subset_size >= len(dataloader.dataset):
        return dataloader
        
    # Create random subset indices
    total_samples = len(dataloader.dataset)
    indices = np.random.choice(total_samples, size=subset_size, replace=False)
    subset_dataset = Subset(dataloader.dataset, indices)
    
    # Create new DataLoader with same parameters but subset data
    return DataLoader(
        subset_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,  # Don't shuffle for consistent evaluation
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate inference model with quantization on validation set")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt file)")
    parser.add_argument("--dataset", choices=["val", "train", "test", "all"], default="val",
                        help="Dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Device to use")
    
    # Quantization parameters
    parser.add_argument("--quant-method", choices=["linear", "log2", "both"], default="both",
                        help="Quantization method to use")
    parser.add_argument("--linear-bits", type=str, default="8,4,3",
                        help="Comma-separated list of linear quantization bit widths")
    parser.add_argument("--log2-levels", type=str, default="8,4,3", 
                        help="Comma-separated list of log2 quantization levels")
    parser.add_argument("--global-percentile", type=float, default=100.0,
                        help="Percentile for global quantization scale (default: 100)")
    
    # Subset evaluation for testing
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Evaluate on subset of data (for testing). None = full dataset")
    
    # Output control
    parser.add_argument("--verbose", action="store_true", help="Show progress bars during evaluation")
    parser.add_argument("--normalize-to-unit", action="store_true", 
                        help="Normalize quantized weights to [-1,+1] range for hardware deployment")
    
    args = parser.parse_args()

    # Load configuration and setup device
    cfg: Dict[str, Any] = load_config(args.config)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Config: {args.config}")
    print(f"Weights: {args.weights}")

    # Create evaluation DataLoader
    train_loader, val_loader, test_loader = make_datasets(cfg, which="all", batch_size=args.batch_size)
    
    if args.dataset == "train":
        eval_loader = train_loader
    elif args.dataset == "val": 
        eval_loader = val_loader
    elif args.dataset == "test":
        eval_loader = test_loader
    else:  # "all"
        # Combine all datasets
        combo = ConcatDataset([train_loader.dataset, val_loader.dataset, test_loader.dataset])
        nw = int(cfg.get("train", {}).get("num_workers", 0))
        pin = bool(cfg.get("train", {}).get("pin_memory", False))
        eval_loader = DataLoader(combo, batch_size=args.batch_size, shuffle=False, 
                                num_workers=nw, pin_memory=pin)

    # Create subset if requested
    if args.subset_size is not None:
        eval_loader = create_subset_loader(eval_loader, args.subset_size)
        print(f"Using subset of {args.subset_size} samples")
    else:
        print(f"Using full dataset: {len(eval_loader.dataset)} samples")

    # Create temporary weights file for quantized models
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize original model using inference wrapper
        print("\n=== Evaluating Original (Float) Model ===")
        original_model = KeywordModel(args.config, args.weights, device)
        
        # Evaluate original model
        original_metrics = original_model.evaluate_dataloader(eval_loader, device, verbose=args.verbose)
        print(f"Original: Acc={original_metrics['accuracy']:.4f} "
              f"P={original_metrics['precision']:.4f} R={original_metrics['recall']:.4f} "
              f"F1={original_metrics['f1']:.4f}")
        
        # Storage for all results
        results: List[Tuple[str, str, Dict[str, float]]] = []
        results.append(("original", "float", original_metrics))
        
        # Parse quantization configurations
        def parse_list(s: str) -> List[str]:
            return [item.strip() for item in s.split(",") if item.strip()]
        
        # Linear quantization evaluation
        if args.quant_method in ["linear", "both"]:
            print("\n=== Evaluating Linear Quantization ===")
            linear_bits_list = parse_list(args.linear_bits)
            
            for bits_str in linear_bits_list:
                bits = int(bits_str)
                print(f"\nLinear {bits}-bit quantization...")
                
                # Apply linear quantization to original model
                model_linear = apply_linear_quantization(
                    original_model.model, 
                    weight_bits=bits,
                    global_percentile=args.global_percentile,
                    normalize_to_unit=args.normalize_to_unit
                )
                
                # Save quantized weights temporarily
                temp_weights = os.path.join(temp_dir, f"linear_{bits}bit.pt")
                torch.save(model_linear.state_dict(), temp_weights)
                
                # Create new inference model with quantized weights
                linear_model = KeywordModel(args.config, temp_weights, device)
                
                # Evaluate quantized model
                linear_metrics = linear_model.evaluate_dataloader(eval_loader, device, verbose=args.verbose)
                results.append(("linear", f"{bits}bit", linear_metrics))
                
                print(f"Linear {bits}-bit: Acc={linear_metrics['accuracy']:.4f} "
                      f"P={linear_metrics['precision']:.4f} R={linear_metrics['recall']:.4f} "
                      f"F1={linear_metrics['f1']:.4f} "
                      f"(Acc drop: {original_metrics['accuracy'] - linear_metrics['accuracy']:.4f})")
        
        # Log2 quantization evaluation  
        if args.quant_method in ["log2", "both"]:
            print("\n=== Evaluating Log2 Quantization ===")
            log2_levels_list = parse_list(args.log2_levels)
            
            for levels_str in log2_levels_list:
                levels = int(levels_str)
                print(f"\nLog2 {levels}-level quantization...")
                
                # Apply log2 quantization to original model
                model_log2 = apply_log2_quantization(
                    original_model.model,
                    num_levels=levels,
                    global_percentile=args.global_percentile,
                    normalize_to_unit=args.normalize_to_unit
                )
                
                # Save quantized weights temporarily
                temp_weights = os.path.join(temp_dir, f"log2_{levels}lvl.pt")
                torch.save(model_log2.state_dict(), temp_weights)
                
                # Create new inference model with quantized weights
                log2_model = KeywordModel(args.config, temp_weights, device)
                
                # Evaluate quantized model
                log2_metrics = log2_model.evaluate_dataloader(eval_loader, device, verbose=args.verbose)
                results.append(("log2", f"{levels}lvl", log2_metrics))
                
                print(f"Log2 {levels}-level: Acc={log2_metrics['accuracy']:.4f} "
                      f"P={log2_metrics['precision']:.4f} R={log2_metrics['recall']:.4f} "
                      f"F1={log2_metrics['f1']:.4f} "
                      f"(Acc drop: {original_metrics['accuracy'] - log2_metrics['accuracy']:.4f})")

    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)

    # Generate and save plots
    print("\n=== Generating Plots ===")
    
    # Determine output directory (model's inference folder)
    # Extract model name from config path or weights path
    config_dir = os.path.dirname(args.config)
    if "experiments" in config_dir:
        model_name = os.path.basename(args.config).replace('.yaml', '')
    else:
        # Fallback to extracting from weights path
        model_name = os.path.basename(os.path.dirname(args.weights))
    
    output_dir = os.path.join(cfg["output"]["plots_dir"], "inference")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Separate results by quantization method
    original_results = [r for r in results if r[0] == "original"]
    linear_results = [r for r in results if r[0] == "linear"]
    log2_results = [r for r in results if r[0] == "log2"]
    
    def plot_metric_comparison(ax, metric_name: str, ylabel: str):
        """Plot comparison of a specific metric across quantization methods."""
        # Original baseline
        if original_results:
            orig_value = original_results[0][2][metric_name]
            ax.axhline(y=orig_value, color='black', linestyle='--', alpha=0.7, label='Original (Float)')
        
        # Linear quantization
        if linear_results:
            linear_labels = [r[1] for r in linear_results]
            linear_values = [r[2][metric_name] for r in linear_results]
            x_linear = range(len(linear_labels))
            ax.plot(x_linear, linear_values, 'o-', color='blue', label='Linear Quant')
            
            # Add bit annotations
            for i, (x, y, label) in enumerate(zip(x_linear, linear_values, linear_labels)):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Log2 quantization  
        if log2_results:
            log2_labels = [r[1] for r in log2_results]
            log2_values = [r[2][metric_name] for r in log2_results]
            x_log2 = range(len(linear_results), len(linear_results) + len(log2_results))
            ax.plot(x_log2, log2_values, 's-', color='red', label='Log2 Quant')
            
            # Add level annotations
            for i, (x, y, label) in enumerate(zip(x_log2, log2_values, log2_labels)):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Format plot
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set x-axis labels
        all_labels = []
        if linear_results:
            all_labels.extend([f"L-{r[1]}" for r in linear_results])
        if log2_results:
            all_labels.extend([f"Log2-{r[1]}" for r in log2_results])
            
        if all_labels:
            ax.set_xticks(range(len(all_labels)))
            ax.set_xticklabels(all_labels, rotation=45)
    
    # Plot each metric
    plot_metric_comparison(ax1, 'accuracy', 'Accuracy')
    ax1.set_title('Classification Accuracy')
    
    plot_metric_comparison(ax2, 'precision', 'Precision')  
    ax2.set_title('Precision (Macro Avg)')
    
    plot_metric_comparison(ax3, 'recall', 'Recall')
    ax3.set_title('Recall (Macro Avg)')
    
    plot_metric_comparison(ax4, 'f1', 'F1 Score')
    ax4.set_title('F1 Score (Macro Avg)')
    
    # Overall title and layout
    dataset_info = f"{args.dataset}"
    if args.subset_size:
        dataset_info += f" (subset: {args.subset_size})"
    
    plt.suptitle(f'Inference Quantization Evaluation - {model_name} - {dataset_info}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"quantization_eval_{model_name}_{args.dataset}"
    if args.subset_size:
        plot_filename += f"_subset{args.subset_size}"
    plot_filename += ".png"
    
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_path}")
    
    # Print summary table
    print(f"\n=== Final Results Summary ===")
    print(f"{'Method':<12} {'Config':<8} {'Accuracy':<8} {'Precision':<9} {'Recall':<8} {'F1':<8} {'Acc Drop':<8}")
    print("-" * 70)
    
    for method, config, metrics in results:
        acc_drop = original_metrics['accuracy'] - metrics['accuracy'] if method != "original" else 0.0
        print(f"{method:<12} {config:<8} {metrics['accuracy']:<8.4f} {metrics['precision']:<9.4f} "
              f"{metrics['recall']:<8.4f} {metrics['f1']:<8.4f} {acc_drop:<8.4f}")


if __name__ == "__main__":
    main()
