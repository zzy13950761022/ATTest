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
# Test fixtures and helper functions for G2 group
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

# ==== BLOCK:CASE_03 START ====
# TC-03: LayerNorm 基本前向传播
@pytest.mark.parametrize("normalized_shape,eps,elementwise_affine,dtype,device,shape", [
    # Base case from test plan: 2D normalized shape
    ([8, 8], 1e-5, True, torch.float32, "cpu", (2, 4, 8, 8)),
    # Parameter extension: 1D normalized shape, no affine, float64
    (8, 1e-6, False, torch.float64, "cpu", (2, 4, 8)),
])
def test_layernorm_forward_basic(normalized_shape, eps, elementwise_affine, dtype, device, shape, set_random_seed):
    """Test basic forward pass of LayerNorm"""
    # Skip CUDA tests if device not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create input tensor
    torch.manual_seed(42)
    input_tensor = torch.randn(*shape, dtype=dtype, device=device)
    
    # Create LayerNorm layer
    ln = LayerNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine
    ).to(device=device, dtype=dtype)
    
    # Forward pass
    output = ln(input_tensor)
    
    # Weak assertions
    # 1. Shape assertion
    assert output.shape == input_tensor.shape, \
        f"Output shape {output.shape} != input shape {input_tensor.shape}"
    
    # 2. Dtype assertion
    assert output.dtype == dtype, \
        f"Output dtype {output.dtype} != expected {dtype}"
    
    # 3. Finite values assertion
    assert torch.all(torch.isfinite(output)), \
        "Output contains NaN or infinite values"
    
    # 4. Basic property: normalized dimensions should have mean ~0 and std ~1
    # Determine which dimensions to normalize over
    if isinstance(normalized_shape, int):
        normalized_dims = (-1,)
        normalized_size = normalized_shape
    else:
        normalized_dims = tuple(range(-len(normalized_shape), 0))
        normalized_size = math.prod(normalized_shape)
    
    # Reshape for statistics calculation
    # Flatten the normalized dimensions
    batch_dims = shape[:len(shape) - len(normalized_dims)]
    batch_size = math.prod(batch_dims) if batch_dims else 1
    
    output_reshaped = output.reshape(batch_size, normalized_size)
    
    # Check statistics for each batch element
    for i in range(batch_size):
        batch_output = output_reshaped[i]
        
        # Mean should be close to 0
        mean_abs = torch.abs(torch.mean(batch_output))
        assert mean_abs < 0.1, f"Batch element {i} mean too large: {mean_abs}"
        
        # Std should be close to 1 (with eps adjustment)
        std = torch.std(batch_output)
        # Allow some tolerance for numerical precision
        assert 0.9 < std < 1.1, f"Batch element {i} std out of range: {std}"
    
    # 5. Check affine parameters if enabled
    if elementwise_affine:
        assert hasattr(ln, 'weight'), "Elementwise affine enabled but weight parameter missing"
        assert hasattr(ln, 'bias'), "Elementwise affine enabled but bias parameter missing"
        
        # Weight and bias should have normalized_shape
        expected_shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        assert ln.weight.shape == expected_shape, \
            f"Weight shape {ln.weight.shape} != expected {expected_shape}"
        assert ln.bias.shape == expected_shape, \
            f"Bias shape {ln.bias.shape} != expected {expected_shape}"
    
    # 6. Compare with functional implementation (weak comparison)
    if elementwise_affine:
        # When elementwise_affine=True, use weight and bias
        try:
            functional_output = F.layer_norm(
                input_tensor, normalized_shape, ln.weight, ln.bias, eps
            )
            # Basic shape check
            assert functional_output.shape == output.shape
        except Exception as e:
            pytest.fail(f"Functional layer_norm failed: {e}")
    else:
        # When elementwise_affine=False, compare directly
        functional_output = F.layer_norm(
            input_tensor, normalized_shape, None, None, eps
        )
        # Weak comparison: just check shapes match
        assert functional_output.shape == output.shape
    
    # 7. Test with different input values
    # Test with all zeros (should produce zeros output)
    zeros_input = torch.zeros(*shape, dtype=dtype, device=device)
    zeros_output = ln(zeros_input)
    
    # For zero input with affine=False, output should be zeros (0/eps = 0)
    if not elementwise_affine:
        assert torch.allclose(zeros_output, torch.zeros_like(zeros_output), atol=1e-7), \
            "Zero input should produce zero output when affine=False"
    
    # Test with all ones (should normalize)
    ones_input = torch.ones(*shape, dtype=dtype, device=device)
    ones_output = ln(ones_input)
    
    # Should still have correct shape and finite values
    assert ones_output.shape == shape
    assert torch.all(torch.isfinite(ones_output))
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_07 START ====
# TC-07: DEFERRED - LayerNorm 参数扩展测试
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# TC-08: DEFERRED - LayerNorm 异常形状测试
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Additional test classes and helper functions can be added here
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====