"""
Analog Weight Error Inference Evaluation

Evaluates quantized models by loading .npz weights and adding random analog errors
to simulate impairments in analog computation hardware (e.g., memristor variations,
ADC/DAC noise, temperature drift).

The noise magnitude is specified as a fraction of the LSB (quantization step size),
which is more realistic for hardware modeling than absolute or weight-range-relative errors.

Usage:
    python -m analysis.eval_inf_weight_analog_error --config CONFIG --weights WEIGHTS.npz --analog-weight-error 0.1,0.5,1.0 [options]
"""

import argparse
import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import load_config
from model.model import DilatedTCN
from train.qat import create_qat_dataloaders
from data_loader.utils import get_num_classes


def compute_global_lsb(npz_weights: Dict[str, np.ndarray]) -> float:
    """
    Compute the global LSB (quantization step size) from loaded quantized weights.
    
    Args:
        npz_weights: Dictionary of quantized weight arrays from all layers
        
    Returns:
        LSB (minimum quantization step) found across all weights
    """
    # Collect all unique weight values from all layers
    all_unique_values = set()
    for weights in npz_weights.values():
        unique_vals = np.unique(weights)
        all_unique_values.update(unique_vals)
    
    # Convert to sorted array
    all_unique_values = np.array(sorted(all_unique_values))
    
    if len(all_unique_values) < 2:
        return 0.0  # Can't determine LSB with fewer than 2 unique values
    
    # Find minimum non-zero difference between consecutive unique values
    diffs = np.diff(all_unique_values)
    nonzero_diffs = diffs[diffs > 1e-10]  # Filter out near-zero differences due to floating point precision
    
    if len(nonzero_diffs) == 0:
        return 0.0
    
    lsb = np.min(nonzero_diffs)
    return lsb


def add_analog_weight_errors(weights: np.ndarray, error_std: float, lsb: float, error_type: str = "gaussian") -> np.ndarray:
    """
    Add analog hardware errors to quantized weights.
    
    Args:
        weights: Quantized weight array
        error_std: Standard deviation of analog errors (as fraction of LSB)
        lsb: Global LSB (quantization step size)
        error_type: Type of error distribution ("gaussian", "uniform", "laplace")
    
    Returns:
        Weights with added analog errors
    """
    if error_std <= 0 or lsb <= 0:
        return weights.copy()
    
    # Generate errors relative to LSB
    error_magnitude = error_std * lsb
    
    if error_type == "gaussian":
        errors = np.random.normal(0, error_magnitude, weights.shape)
    elif error_type == "uniform":
        # Uniform distribution with same std as Gaussian
        # For uniform on [-a, a]: std = a/sqrt(3), so a = std*sqrt(3)
        half_width = error_magnitude * np.sqrt(3)
        errors = np.random.uniform(-half_width, half_width, weights.shape)
    elif error_type == "laplace":
        # Laplace distribution with same std as Gaussian
        # For Laplace with scale b: std = b*sqrt(2), so b = std/sqrt(2)
        scale = error_magnitude / np.sqrt(2)
        errors = np.random.laplace(0, scale, weights.shape)
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    
    return weights + errors


def evaluate_with_analog_errors(
    model: nn.Module,
    npz_weights: Dict[str, np.ndarray],
    loader,
    device: torch.device,
    error_std: float,
    error_type: str = "gaussian",
    apply_constraints: bool = True,
    weight_min: float = -1.0,
    weight_max: float = 1.0,
    act_min: float = 0.0,
    act_max: float = 1024.0,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate model with analog weight errors applied to quantized weights.
    
    Args:
        model: PyTorch model
        npz_weights: Dictionary of quantized weights from NPZ file
        loader: DataLoader for evaluation
        device: Device to run evaluation on
        error_std: Standard deviation of analog errors (as fraction of LSB)
        error_type: Type of error distribution ("gaussian", "uniform", "laplace")
        apply_constraints: Whether to apply hardware constraints during evaluation
        weight_min: Minimum weight value for hardware constraints (from config)
        weight_max: Maximum weight value for hardware constraints (from config)
        act_min: Minimum activation value for hardware constraints (from config)
        act_max: Maximum activation value for hardware constraints (from config)
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with evaluation metrics (accuracy, precision, recall, f1)
    """
    # Compute global LSB from all quantized weights
    global_lsb = compute_global_lsb(npz_weights)
    
    # Apply analog errors to quantized weights
    noisy_state_dict = {}
    total_weights = 0
    
    for layer_name, weights in npz_weights.items():
        noisy_weights = add_analog_weight_errors(weights, error_std, global_lsb, error_type)
        
        # Clip weights to hardware constraints after adding analog errors
        if apply_constraints:
            noisy_weights = np.clip(noisy_weights, weight_min, weight_max)
        
        weight_tensor = torch.from_numpy(noisy_weights).float()
        noisy_state_dict[layer_name] = weight_tensor
        
        # Track error statistics
        total_weights += weights.size
    
    # Load noisy weights into model
    model.load_state_dict(noisy_state_dict, strict=False)
    model.to(device).eval()
    
    # Report error statistics
    if verbose and error_std > 0:
        error_magnitude = error_std * global_lsb
        print(f"[ANALOG ERROR] Applied {error_type} noise: {error_std:.3f}×LSB "
              f"(LSB={global_lsb:.6f}, noise_std={error_magnitude:.6f})")
    elif verbose:
        print(f"[ANALOG ERROR] No noise applied (global LSB={global_lsb:.6f})")
    
    # Apply activation constraint hooks if requested
    hooks = []
    if apply_constraints:
        def activation_constraint_hook(module, input, output):
            return torch.clamp(output, act_min, act_max)
        
        # Apply hooks to modules with 'act' in name (consistent with QAT training)
        for name, module in model.named_modules():
            if 'act' in name:
                hooks.append(module.register_forward_hook(activation_constraint_hook))
    
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    # Set up progress bar
    iterator = tqdm(loader, desc=f"Evaluating (error={error_std:.2f}×LSB)") if verbose else loader
    
    with torch.no_grad():
        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Apply per-sample input scaling for hardware constraints (consistent with PTQ/QAT)
            if apply_constraints:
                # Per-sample normalization to [act_min, act_max] using config limits
                # Handles 2D (batch_size, features) and 3D (batch_size, features, time)
                dims = tuple(range(1, batch_x.dim()))
                min_vals = batch_x.amin(dim=dims, keepdim=True)
                max_vals = batch_x.amax(dim=dims, keepdim=True)
                denom = (max_vals - min_vals).clamp(min=1e-8)
                # Scale to config activation range per sample
                batch_x = (batch_x - min_vals) / denom * (act_max - act_min) + act_min
            
            # Forward pass with noisy weights
            logits = model(batch_x)
            predictions = torch.argmax(logits, dim=1)
            
            # Accumulate results
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Clean up constraint hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate metrics
    accuracy = total_correct / total_samples
    
    # Calculate per-class precision, recall, F1
    from sklearn.metrics import classification_report
    report = classification_report(all_targets, all_predictions, output_dict=True, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1': report['macro avg']['f1-score']
    }


def plot_error_robustness(
    results: List[Tuple[float, Dict[str, float]]],
    output_path: str,
    title_suffix: str = ""
):
    """
    Plot accuracy degradation vs analog weight error.
    
    Args:
        results: List of (error_std, metrics) tuples
        output_path: Path to save plot
        title_suffix: Additional text for plot title
    """
    error_stds = [r[0] for r in results]
    accuracies = [r[1]['accuracy'] for r in results]
    precisions = [r[1]['precision'] for r in results]
    recalls = [r[1]['recall'] for r in results]
    f1_scores = [r[1]['f1'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy
    ax1.plot(error_stds, accuracies, 'o-', color='blue', linewidth=2, markersize=6)
    ax1.set_xlabel('Analog Weight Error (× LSB)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Classification Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # Precision
    ax2.plot(error_stds, precisions, 'o-', color='green', linewidth=2, markersize=6)
    ax2.set_xlabel('Analog Weight Error (× LSB)')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision (Macro Avg)')
    ax2.grid(True, alpha=0.3)
    
    # Recall
    ax3.plot(error_stds, recalls, 'o-', color='orange', linewidth=2, markersize=6)
    ax3.set_xlabel('Analog Weight Error (× LSB)')
    ax3.set_ylabel('Recall')
    ax3.set_title('Recall (Macro Avg)')
    ax3.grid(True, alpha=0.3)
    
    # F1 Score
    ax4.plot(error_stds, f1_scores, 'o-', color='red', linewidth=2, markersize=6)
    ax4.set_xlabel('Analog Weight Error (× LSB)')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score (Macro Avg)')
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle(f'Analog Weight Error Robustness{title_suffix}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[PLOT] Saved robustness plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized models with analog weight errors")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--weights", required=True, help="Path to quantized weights (.npz file)")
    
    # Analog error parameters
    parser.add_argument("--analog-weight-error", type=str, default="0.0,0.1,0.3,0.5,1.0,2.0",
                       help="Comma-separated list of analog weight error std values (as fraction of LSB)")
    parser.add_argument("--error-type", choices=["gaussian", "uniform", "laplace"], default="gaussian",
                       help="Type of analog error distribution (default: gaussian)")
    parser.add_argument("--num-trials", type=int, default=1,
                       help="Number of trials to average over for each error level (default: 1)")
    
    # Hardware constraints
    parser.add_argument("--no-constraints", action="store_true",
                       help="Disable hardware constraints during evaluation")
    
    # Evaluation parameters
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation (default: 32)")
    parser.add_argument("--dataset", choices=["train", "val", "test"], default="val",
                       help="Which dataset to evaluate on (default: val)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show progress bar and detailed output")
    
    # Output control
    parser.add_argument("--save-results", action="store_true",
                       help="Save detailed results to file")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract hardware constraints from train config section (consistent with PTQ/QAT)
    train_cfg = cfg.get("train", {})
    weight_min = float(train_cfg.get("weight_min", -1.0))
    weight_max = float(train_cfg.get("weight_max", 1.0))
    act_min = float(train_cfg.get("act_min", 0.0))
    act_max = float(train_cfg.get("act_max", 1024.0))
    
    if args.verbose:
        print(f"Using device: {device}")
        print(f"Config: {args.config}")
        print(f"Quantized weights: {args.weights}")
        print(f"Weight limits from train config: [{weight_min}, {weight_max}]")
        print(f"Activation limits from train config: [{act_min}, {act_max}]")
        print(f"Error type: {args.error_type}")
        print(f"Number of trials: {args.num_trials}")
        print(f"Hardware constraints: {'Enabled' if not args.no_constraints else 'Disabled'}")
    
    # Parse error levels
    error_stds = [float(x.strip()) for x in args.analog_weight_error.split(",") if x.strip()]
    if args.verbose:
        print(f"Error levels: {error_stds}")
    
    # Create dataset
    train_loader, val_loader, test_loader = create_qat_dataloaders(
        cfg, args.batch_size, num_workers=0, pin_memory=False, 
        persistent_workers=False, prefetch_factor=2
    )
    
    # Select dataset
    if args.dataset == "train":
        eval_loader = train_loader
        dataset_name = "Training"
    elif args.dataset == "val":
        eval_loader = val_loader
        dataset_name = "Validation"
    else:  # test
        eval_loader = test_loader
        dataset_name = "Test"
    
    if args.verbose:
        print(f"Evaluating on {dataset_name} set: {len(eval_loader.dataset)} samples")
    
    # Load quantized weights from NPZ file
    npz_weights = np.load(args.weights, allow_pickle=True)
    npz_dict = {layer_name: npz_weights[layer_name] for layer_name in npz_weights.files}
    
    if args.verbose:
        print(f"Loaded {len(npz_dict)} quantized weight tensors")
        
        # Show quantization info for first layer
        first_layer_name = list(npz_dict.keys())[0]
        first_weights = npz_dict[first_layer_name]
        unique_vals = len(np.unique(first_weights))
        global_lsb = compute_global_lsb(npz_dict)
        print(f"Example: {first_layer_name} has {unique_vals} unique values")
        print(f"Global LSB (quantization step): {global_lsb:.6f}")
    
    # Create model template
    model_template = DilatedTCN.from_config(cfg)
    
    # Evaluate across error levels
    results = []
    apply_constraints = not args.no_constraints
    
    print(f"\n=== Evaluating Analog Weight Error Robustness ===")
    
    for error_std in error_stds:
        if args.verbose:
            print(f"\n--- Error std: {error_std:.4f} ---")
        
        # Average over multiple trials
        trial_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for trial in range(args.num_trials):
            # Set random seed for reproducible noise (but different per trial)
            np.random.seed(42 + trial)
            
            # Create fresh model for this trial
            model = DilatedTCN.from_config(cfg)
            
            # Evaluate with analog errors
            metrics = evaluate_with_analog_errors(
                model=model,
                npz_weights=npz_dict,
                loader=eval_loader,
                device=device,
                error_std=error_std,
                error_type=args.error_type,
                apply_constraints=apply_constraints,
                weight_min=weight_min,
                weight_max=weight_max,
                act_min=act_min,
                act_max=act_max,
                verbose=args.verbose and args.num_trials == 1
            )
            
            # Collect trial results
            for key in trial_results:
                trial_results[key].append(metrics[key])
        
        # Average across trials
        avg_metrics = {
            key: np.mean(values) for key, values in trial_results.items()
        }
        std_metrics = {
            key: np.std(values) for key, values in trial_results.items()
        } if args.num_trials > 1 else None
        
        results.append((error_std, avg_metrics))
        
        # Print results
        if args.num_trials > 1 and std_metrics:
            print(f"Error std {error_std:.4f}: "
                  f"Acc={avg_metrics['accuracy']:.4f}±{std_metrics['accuracy']:.4f} "
                  f"P={avg_metrics['precision']:.4f}±{std_metrics['precision']:.4f} "
                  f"R={avg_metrics['recall']:.4f}±{std_metrics['recall']:.4f} "
                  f"F1={avg_metrics['f1']:.4f}±{std_metrics['f1']:.4f}")
        else:
            print(f"Error std {error_std:.4f}: "
                  f"Acc={avg_metrics['accuracy']:.4f} "
                  f"P={avg_metrics['precision']:.4f} "
                  f"R={avg_metrics['recall']:.4f} "
                  f"F1={avg_metrics['f1']:.4f}")
    
    # Generate plots
    print(f"\n=== Generating Plots ===")
    
    # Determine output directory
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    weights_name = os.path.splitext(os.path.basename(args.weights))[0]
    
    output_dir = os.path.join(cfg.get("output", {}).get("plots_dir", "plots"), "analog_error")
    plot_filename = f"analog_weight_error_{config_name}_{weights_name}_{args.error_type}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    title_suffix = f" ({config_name}, {args.error_type} noise)"
    plot_error_robustness(results, plot_path, title_suffix)
    
    # Save detailed results if requested
    if args.save_results:
        results_dir = os.path.join(output_dir, "detailed_results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_filename = f"analog_error_results_{config_name}_{weights_name}_{args.error_type}.txt"
        results_path = os.path.join(results_dir, results_filename)
        
        with open(results_path, 'w') as f:
            f.write(f"Analog Weight Error Evaluation Results\n")
            f.write(f"=====================================\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Weights: {args.weights}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Error type: {args.error_type}\n")
            f.write(f"Number of trials: {args.num_trials}\n")
            f.write(f"Hardware constraints: {'Enabled' if apply_constraints else 'Disabled'}\n")
            if apply_constraints:
                f.write(f"  Weight range: [{weight_min}, {weight_max}]\n")
                f.write(f"  Activation range: [{act_min}, {act_max}]\n")
            f.write(f"Device: {device}\n\n")
            
            f.write(f"Results (error std as fraction of LSB):\n")
            f.write(f"{'Error(×LSB)':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1_Score':<10}\n")
            f.write("-" * 65 + "\n")
            
            for error_std, metrics in results:
                f.write(f"{error_std:<12.3f} {metrics['accuracy']:<10.4f} "
                       f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                       f"{metrics['f1']:<10.4f}\n")
        
        print(f"[RESULTS] Saved detailed results: {results_path}")
    
    print(f"\n=== Summary ===")
    baseline_acc = results[0][1]['accuracy'] if results else 0
    print(f"Baseline accuracy (no error): {baseline_acc:.4f}")
    
    if len(results) > 1:
        worst_acc = min(r[1]['accuracy'] for r in results[1:])
        worst_error = next(r[0] for r in results if r[1]['accuracy'] == worst_acc)
        print(f"Worst accuracy: {worst_acc:.4f} at {worst_error:.2f}×LSB")
        print(f"Max degradation: {baseline_acc - worst_acc:.4f} ({100*(baseline_acc - worst_acc)/baseline_acc:.2f}%)")


if __name__ == "__main__":
    main()