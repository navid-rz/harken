"""
Model export utilities for deployment.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple


def export_to_torchscript(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str,
    optimize: bool = True
) -> None:
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        sample_input: Example input tensor for tracing
        output_path: Path to save .pt file
        optimize: Apply optimization passes
    """
    model.eval()
    
    # Trace model
    traced = torch.jit.trace(model, sample_input)
    
    if optimize:
        traced = torch.jit.optimize_for_inference(traced)
    
    # Save
    torch.jit.save(traced, output_path)
    print(f"[OK] TorchScript model saved to {output_path}")
    
    # Verify
    loaded = torch.jit.load(output_path)
    with torch.no_grad():
        orig_out = model(sample_input)
        jit_out = loaded(sample_input)
        diff = (orig_out - jit_out).abs().max().item()
        print(f"[INFO] Max output difference: {diff:.2e}")


def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str,
    opset_version: int = 14,
    dynamic_axes: Optional[dict] = None
) -> None:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        sample_input: Example input tensor
        output_path: Path to save .onnx file
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes specification (e.g., {"input": {0: "batch"}})
    """
    model.eval()
    
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch"},
            "output": {0: "batch"}
        }
    
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    print(f"[OK] ONNX model saved to {output_path}")
    
    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort
        import numpy as np
        
        sess = ort.InferenceSession(output_path)
        ort_input = {sess.get_inputs()[0].name: sample_input.cpu().numpy()}
        ort_out = sess.run(None, ort_input)[0]
        
        with torch.no_grad():
            torch_out = model(sample_input).cpu().numpy()
        
        diff = np.abs(torch_out - ort_out).max()
        print(f"[INFO] Max ONNX output difference: {diff:.2e}")
    except ImportError:
        print("[WARN] onnxruntime not available, skipping verification")


def get_model_info(model: nn.Module) -> dict:
    """Get model architecture info."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    }
