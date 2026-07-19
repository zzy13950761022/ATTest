import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import (
    LocalResponseNorm,
    CrossMapLRN2d,
    LayerNorm,
    GroupNorm
)

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G3 group
@pytest.fixture(scope="function")
def set_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    return 42

def assert_tensor_properties(tensor, expected_shape=None, expected_dtype=None, 
                           expected_device=None, name=""):
    """Helper to assert tensor properties"""
    assert torch.is_tensor(tensor), f"{name}: Output is not a tensor"
    assert torch.all(torch.isfinite(tensor)), f"{name}: Tensor contains NaN or Inf"
    
    if expected_shape is not None:
        assert tensor.shape == expected_shape, \
            f"{name}: Shape mismatch: {tensor.shape} != {expected_shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, \
            f"{name}: Dtype mismatch: {tensor.dtype} != {expected_dtype}"
    
    if expected_device is not None:
        assert tensor.device == expected_device, \
            f"{name}: Device mismatch: {tensor.device} != {expected_device}"
    
    return True
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: LocalResponseNorm 基本前向传播
@pytest.mark.parametrize("size,alpha,beta,k,dtype,device,shape", [
    # Base case from test plan
    (5, 1e-4, 0.75, 1.0, torch.float32, "cpu", (2, 16, 8, 8)),
])
def test_localresponsenorm_forward_basic(size, alpha, beta, k, dtype, device, shape, set_random_seed):
    """Test basic forward pass of LocalResponseNorm"""
    # Skip CUDA tests if device not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create input tensor
    torch.manual_seed(42)
    input_tensor = torch.randn(*shape, dtype=dtype, device=device)
    
    # Create LocalResponseNorm layer
    lrn = LocalResponseNorm(
        size=size,
        alpha=alpha,
        beta=beta,
        k=k
    ).to(device=device)
    
    # Forward pass
    output = lrn(input_tensor)
    
    # Weak assertions
    # 1. Shape assertion
    assert output.shape == input_tensor.shape, \
        f"Output shape {output.shape} != input shape {input_tensor.shape}"
    
    # 2. Dtype assertion - LRN doesn't change dtype
    assert output.dtype == input_tensor.dtype, \
        f"Output dtype {output.dtype} != input dtype {input_tensor.dtype}"
    
    # 3. Finite values assertion
    assert torch.all(torch.isfinite(output)), \
        "Output contains NaN or infinite values"
    
    # 4. Basic property: output should have same sign as input
    # LRN is a pointwise normalization, so sign should be preserved
    sign_preserved = torch.all((input_tensor * output) >= 0)
    assert sign_preserved, "LocalResponseNorm should preserve sign of input"
    
    # 5. Basic property: output magnitude should be reduced or similar
    # LRN reduces magnitude based on neighboring channels
    input_norm = torch.norm(input_tensor)
    output_norm = torch.norm(output)
    
    # Output norm should not be larger than input norm (normalization reduces)
    # Allow small numerical differences
    assert output_norm <= input_norm * 1.1, \
        f"Output norm {output_norm} > input norm {input_norm} * 1.1"
    
    # 6. Check layer parameters
    assert lrn.size == size, f"Layer size {lrn.size} != expected {size}"
    assert abs(lrn.alpha - alpha) < 1e-10, f"Layer alpha {lrn.alpha} != expected {alpha}"
    assert abs(lrn.beta - beta) < 1e-10, f"Layer beta {lrn.beta} != expected {beta}"
    assert abs(lrn.k - k) < 1e-10, f"Layer k {lrn.k} != expected {k}"
    
    # 7. Compare with functional implementation (weak comparison)
    try:
        functional_output = F.local_response_norm(
            input_tensor, size, alpha, beta, k
        )
        # Basic shape check
        assert functional_output.shape == output.shape
        
        # Weak comparison: check they're close within reasonable tolerance
        # LRN can have numerical differences due to implementation
        diff_norm = torch.norm(functional_output - output)
        rel_diff = diff_norm / (torch.norm(output) + 1e-10)
        assert rel_diff < 0.01, \
            f"Functional and layer outputs differ significantly: rel_diff={rel_diff}"
    except Exception as e:
        pytest.fail(f"Functional local_response_norm failed: {e}")
    
    # 8. Test edge cases
    
    # Test with all positive values
    positive_input = torch.abs(input_tensor) + 0.1  # Ensure all positive
    positive_output = lrn(positive_input)
    assert torch.all(positive_output >= 0), "All positive input should produce non-negative output"
    
    # Test with small size (size=1 means only self-channel)
    if shape[1] >= 3:  # Need at least 3 channels for size=1 to make sense
        lrn_small = LocalResponseNorm(size=1, alpha=alpha, beta=beta, k=k).to(device=device)
        small_output = lrn_small(input_tensor)
        assert small_output.shape == shape
        
        # With size=1, normalization should be minimal
        # Check that output is close to input/(k^beta) when only considering self
        expected_small = input_tensor / (k**beta)
        diff = torch.norm(small_output - expected_small) / torch.norm(expected_small)
        # Allow some difference due to implementation details
        assert diff < 0.1, f"Size=1 LRN differs from expected: diff={diff}"
    
    # 9. Test parameter boundaries
    
    # Test with different alpha values
    for test_alpha in [1e-5, 1e-3, 1e-2]:
        lrn_alpha = LocalResponseNorm(size=size, alpha=test_alpha, beta=beta, k=k).to(device=device)
        output_alpha = lrn_alpha(input_tensor)
        assert output_alpha.shape == shape
        assert torch.all(torch.isfinite(output_alpha))
    
    # Test with different beta values
    for test_beta in [0.5, 0.75, 1.0]:
        lrn_beta = LocalResponseNorm(size=size, alpha=alpha, beta=test_beta, k=k).to(device=device)
        output_beta = lrn_beta(input_tensor)
        assert output_beta.shape == shape
        assert torch.all(torch.isfinite(output_beta))
    
    # Test with different k values
    for test_k in [0.5, 1.0, 2.0]:
        lrn_k = LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=test_k).to(device=device)
        output_k = lrn_k(input_tensor)
        assert output_k.shape == shape
        assert torch.all(torch.isfinite(output_k))
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_09 START ====
# TC-09: DEFERRED - CrossMapLRN2d 基本功能测试
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:CASE_10 START ====
# TC-10: DEFERRED - LocalResponseNorm 边界值测试
# ==== BLOCK:CASE_10 END ====

# ==== BLOCK:FOOTER START ====
# Additional test classes and helper functions can be added here
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====