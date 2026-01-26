"""
Test BatchNorm folding to verify correctness.

This script:
1. Creates a simple Conv→BN→ReLU model
2. Runs inference on random input
3. Folds BN into Conv
4. Runs inference again and compares outputs

Expected: Outputs should be identical (within floating point error)
"""
import torch
import torch.nn as nn
from train.utils import fold_batchnorm


class SimpleConvBN(nn.Module):
    """Test model with Conv→BN→ReLU pattern"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(10, 20, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(20)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(20, 30, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(30)
        self.act2 = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        return out


def test_bn_folding():
    """Test that BN folding preserves model output"""
    print("[TEST] Creating model with BatchNorm...")
    model = SimpleConvBN()
    model.eval()
    
    # Random input
    torch.manual_seed(42)
    x = torch.randn(2, 10, 50)
    
    # Run inference before folding
    print("[TEST] Running inference before folding...")
    with torch.no_grad():
        output_before = model(x)
    
    # Fold BN
    print("[TEST] Folding BatchNorm layers...")
    model_folded = fold_batchnorm(model)
    
    # Run inference after folding
    print("[TEST] Running inference after folding...")
    with torch.no_grad():
        output_after = model_folded(x)
    
    # Compare outputs
    diff = (output_before - output_after).abs().max().item()
    print(f"[TEST] Max absolute difference: {diff:.2e}")
    
    if diff < 1e-5:
        print("[PASS] BatchNorm folding preserves outputs ✓")
    else:
        print(f"[FAIL] Outputs differ by {diff:.2e} (threshold: 1e-5)")
    
    # Verify BN layers are replaced with Identity
    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in model_folded.modules())
    has_identity = any(isinstance(m, nn.Identity) for m in model_folded.modules())
    
    print(f"[TEST] Has BatchNorm after folding: {has_bn}")
    print(f"[TEST] Has Identity placeholders: {has_identity}")
    
    if not has_bn:
        print("[PASS] All BatchNorm layers removed ✓")
    else:
        print("[FAIL] Some BatchNorm layers still present")


if __name__ == "__main__":
    test_bn_folding()
