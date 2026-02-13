"""
QAT Model Inference Evaluation

Evaluates QAT-trained models by loading the final quantized deployment weights
from NPZ files. These weights are already quantized to their final hardware
values, so no additional quantization hooks are needed.

Usage:
    python -m analysis.eval_inf_qat --config CONFIG --weights WEIGHTS.npz [options]
"""

import argparse
import os
import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from tqdm import tqdm

from config import load_config
from model.model import DilatedTCN
from train.qat import create_qat_dataloaders
from data_loader.utils import get_num_classes


def evaluate_quantized_model(
    model: nn.Module, 
    loader, 
    device: torch.device, 
    apply_constraints: bool = True,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate model with pre-quantized deployment weights.
    
    Args:
        model: Model loaded with pre-quantized weights from NPZ file
        loader: DataLoader for evaluation
        device: Device to run evaluation on
        apply_constraints: Whether to apply hardware constraints during evaluation
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    # Set up progress bar
    iterator = tqdm(loader, desc="Evaluating quantized model") if verbose else loader
    
    # Apply activation constraint hooks if requested
    hooks = []
    if apply_constraints:
        def activation_constraint_hook(module, input, output):
            return torch.clamp(output, 0.0, 1024.0)
        
        # Apply hooks to modules with 'act' in name (consistent with training)
        for name, module in model.named_modules():
            if 'act' in name:
                hooks.append(module.register_forward_hook(activation_constraint_hook))
    
    with torch.no_grad():
        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Apply input scaling for hardware constraints if requested
            if apply_constraints:
                batch_min = batch_x.min()
                batch_max = batch_x.max()
                if batch_max > batch_min:
                    batch_x = (batch_x - batch_min) / (batch_max - batch_min) * 1024.0
            
            # Forward pass (quantization happens via hooks)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate QAT-trained models with quantized weights")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--weights", required=True, help="Path to quantized deployment weights (.npz file)")
    # Note: Weight parameters not needed since weights are pre-quantized
    parser.add_argument("--act-max", type=float, default=1024.0,
                       help="Maximum activation value for input scaling (default: 1024.0)")
    parser.add_argument("--no-constraints", action="store_true",
                       help="Disable hardware constraints during evaluation")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation (default: 32)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show progress bar and detailed output")
    parser.add_argument("--dataset", choices=["train", "val", "test"], default="val",
                       help="Which dataset to evaluate on (default: val)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.verbose:
        print(f"Using device: {device}")
        print(f"Config: {args.config}")
        print(f"Quantized weights: {args.weights}")
        print(f"Hardware constraints: {'Enabled' if not args.no_constraints else 'Disabled'}")
    
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
    
    num_classes = get_num_classes(cfg)
    
    if args.verbose:
        print(f"Evaluating on {dataset_name} set: {len(eval_loader.dataset)} samples")
    
    # Create model and load pre-quantized weights from NPZ
    model = DilatedTCN.from_config(cfg)
    
    if args.verbose:
        print(f"Loading pre-quantized weights from NPZ file...")
    
    # Load quantized weights from NPZ file
    npz_weights = np.load(args.weights, allow_pickle=True)
    
    # Convert numpy arrays to PyTorch tensors and load into model
    state_dict = {}
    for layer_name in npz_weights.files:
        # Convert numpy array to PyTorch tensor
        weight_tensor = torch.from_numpy(npz_weights[layer_name]).float()
        state_dict[layer_name] = weight_tensor
    
    model.load_state_dict(state_dict, strict=False)  # strict=False in case of missing bias terms
    model.to(device)
    
    if args.verbose:
        total_params = sum(p.numel() for p in model.parameters())
        num_quantized = len(state_dict)
        print(f"Model loaded with {total_params:,} parameters")
        print(f"Loaded {num_quantized} pre-quantized weight tensors")
        
        # Show quantization info for first layer
        first_layer_name = list(state_dict.keys())[0]
        first_weights = state_dict[first_layer_name]
        unique_vals = len(torch.unique(first_weights))
        print(f"Example: {first_layer_name} has {unique_vals} unique values (quantized)")
    
    if args.verbose:
        print(f"\n=== Evaluating Pre-Quantized Model ===")
    
    # Evaluate model (weights are already quantized, no hooks needed)
    apply_constraints = not args.no_constraints
    results = evaluate_quantized_model(model, eval_loader, device, apply_constraints, args.verbose)
    
    # Display results
    constraint_text = "with constraints" if apply_constraints else "no constraints"
    print(f"\n=== Pre-Quantized Model Results ({constraint_text}) ===")
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    
    # Save results if requested
    if args.verbose:
        output_dir = os.path.join(cfg.get("output", {}).get("plots_dir", "plots"), "qat", "evaluation")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to file
        results_file = os.path.join(output_dir, f"qat_quantized_eval_{args.dataset}_{constraint_text.replace(' ', '_')}.txt")
        with open(results_file, 'w') as f:
            f.write(f"Pre-Quantized Model Evaluation Results\n")
            f.write(f"======================================\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Quantized weights: {args.weights}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Weights: Pre-quantized from QAT training\n")
            f.write(f"Constraints: {constraint_text}\n")
            f.write(f"Device: {device}\n\n")
            f.write(f"Results:\n")
            f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall:    {results['recall']:.4f}\n")
            f.write(f"F1-Score:  {results['f1']:.4f}\n")
        
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()