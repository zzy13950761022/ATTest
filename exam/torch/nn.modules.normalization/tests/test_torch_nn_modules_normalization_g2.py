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
        normalized_shape_tuple = (normalized_shape,)
    else:
        normalized_dims = tuple(range(-len(normalized_shape), 0))
        normalized_size = math.prod(normalized_shape)
        normalized_shape_tuple = tuple(normalized_shape)
    
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
        
        # Weight and bias should have normalized_shape as tuple
        # LayerNorm internally converts normalized_shape to tuple
        expected_shape = normalized_shape_tuple
        assert ln.weight.shape == expected_shape, \
            f"Weight shape {ln.weight.shape} != expected {expected_shape}"
        assert ln.bias.shape == expected_shape, \
            f"Bias shape {ln.bias.shape} != expected {expected_shape}"
    
    # 6. Compare with functional implementation (weak comparison)
    # Convert normalized_shape to list for F.layer_norm
    if isinstance(normalized_shape, int):
        normalized_shape_list = [normalized_shape]
    else:
        normalized_shape_list = list(normalized_shape)
    
    if elementwise_affine:
        # When elementwise_affine=True, use weight and bias
        try:
            functional_output = F.layer_norm(
                input_tensor, normalized_shape_list, ln.weight, ln.bias, eps
            )
            # Basic shape check
            assert functional_output.shape == output.shape, \
                f"Functional output shape {functional_output.shape} != layer output shape {output.shape}"
        except Exception as e:
            pytest.fail(f"Functional layer_norm failed: {e}")
    else:
        # When elementwise_affine=False, compare directly
        functional_output = F.layer_norm(
            input_tensor, normalized_shape_list, None, None, eps
        )
        # Weak comparison: just check shapes match
        assert functional_output.shape == output.shape, \
            f"Functional output shape {functional_output.shape} != layer output shape {output.shape}"
    
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
# TC-07: LayerNorm 参数扩展测试
@pytest.mark.parametrize("normalized_shape,eps,elementwise_affine,dtype,device,shape", [
    # Test various normalized shapes
    (16, 1e-5, True, torch.float32, "cpu", (2, 4, 16)),  # 1D normalization
    ([4, 4], 1e-5, True, torch.float32, "cpu", (2, 8, 4, 4)),  # 2D normalization
    ([2, 2, 2], 1e-5, True, torch.float32, "cpu", (2, 4, 2, 2, 2)),  # 3D normalization
    # Test with no affine
    ([8, 8], 1e-6, False, torch.float64, "cpu", (4, 2, 8, 8)),  # No affine, float64
    # Test with different eps values
    ([4, 4, 4], 1e-7, True, torch.float32, "cpu", (1, 2, 4, 4, 4)),  # Small eps
    ([16], 1e-3, True, torch.float32, "cpu", (8, 4, 16)),  # Large eps
])
def test_layernorm_parameter_extensions(normalized_shape, eps, elementwise_affine, dtype, device, shape, set_random_seed):
    """Test LayerNorm with various parameter extensions"""
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
        normalized_shape_tuple = (normalized_shape,)
    else:
        normalized_dims = tuple(range(-len(normalized_shape), 0))
        normalized_size = math.prod(normalized_shape)
        normalized_shape_tuple = tuple(normalized_shape)
    
    # Reshape for statistics calculation
    batch_dims = shape[:len(shape) - len(normalized_dims)]
    batch_size = math.prod(batch_dims) if batch_dims else 1
    
    output_reshaped = output.reshape(batch_size, normalized_size)
    
    # Check statistics for each batch element
    for i in range(batch_size):
        batch_output = output_reshaped[i]
        
        # Mean should be close to 0
        mean_abs = torch.abs(torch.mean(batch_output))
        assert mean_abs < 0.2, f"Batch element {i} mean too large: {mean_abs}"
        
        # Std should be close to 1 (with eps adjustment)
        std = torch.std(batch_output)
        # Allow tolerance based on eps
        eps_tolerance = max(eps * 10, 0.2)
        assert 1.0 - eps_tolerance < std < 1.0 + eps_tolerance, \
            f"Batch element {i} std out of range: {std} (eps={eps})"
    
    # 5. Check affine parameters if enabled
    if elementwise_affine:
        assert hasattr(ln, 'weight'), "Elementwise affine enabled but weight parameter missing"
        assert hasattr(ln, 'bias'), "Elementwise affine enabled but bias parameter missing"
        
        # Weight and bias should have normalized_shape as tuple
        expected_shape = normalized_shape_tuple
        assert ln.weight.shape == expected_shape, \
            f"Weight shape {ln.weight.shape} != expected {expected_shape}"
        assert ln.bias.shape == expected_shape, \
            f"Bias shape {ln.bias.shape} != expected {expected_shape}"
    
    # 6. Test with extreme input values
    # Test with very small values
    small_input = torch.randn(*shape, dtype=dtype, device=device) * 1e-10
    small_output = ln(small_input)
    assert small_output.shape == shape
    assert torch.all(torch.isfinite(small_output))
    
    # Test with very large values
    large_input = torch.randn(*shape, dtype=dtype, device=device) * 1e10
    large_output = ln(large_input)
    assert large_output.shape == shape
    assert torch.all(torch.isfinite(large_output))
    
    # 7. Test training/eval mode consistency
    ln.train()
    train_output = ln(input_tensor)
    
    ln.eval()
    eval_output = ln(input_tensor)
    
    # LayerNorm has no train/eval mode differences (no running statistics)
    diff = torch.norm(train_output - eval_output) / torch.norm(eval_output)
    assert diff < 1e-7, f"Train/eval mode outputs differ: diff={diff}"
    
    # 8. Test parameter access
    params = list(ln.parameters())
    if elementwise_affine:
        assert len(params) == 2, f"Expected 2 parameters with elementwise_affine=True, got {len(params)}"
        assert params[0].shape == normalized_shape_tuple
        assert params[1].shape == normalized_shape_tuple
    else:
        assert len(params) == 0, f"Expected 0 parameters with elementwise_affine=False, got {len(params)}"
    
    # 9. Test with different input statistics
    # Create input with specific mean and variance
    target_mean = 5.0
    target_std = 2.0
    custom_input = torch.randn(*shape, dtype=dtype, device=device) * target_std + target_mean
    
    custom_output = ln(custom_input)
    assert custom_output.shape == shape
    
    # After normalization, mean should be close to 0, std close to 1
    custom_reshaped = custom_output.reshape(batch_size, normalized_size)
    for i in range(batch_size):
        batch_custom = custom_reshaped[i]
        custom_mean = torch.mean(batch_custom)
        custom_std = torch.std(batch_custom)
        
        assert abs(custom_mean) < 0.2, f"Custom input mean not normalized: {custom_mean}"
        assert 0.8 < custom_std < 1.2, f"Custom input std not normalized: {custom_std}"
    
    # 10. Test functional equivalence
    # Convert normalized_shape to list for F.layer_norm
    if isinstance(normalized_shape, int):
        normalized_shape_list = [normalized_shape]
    else:
        normalized_shape_list = list(normalized_shape)
    
    if elementwise_affine:
        functional_output = F.layer_norm(
            input_tensor, normalized_shape_list, ln.weight, ln.bias, eps
        )
    else:
        functional_output = F.layer_norm(
            input_tensor, normalized_shape_list, None, None, eps
        )
    
    # Check they're close
    diff = torch.norm(functional_output - output) / torch.norm(output)
    assert diff < 1e-5, f"Functional and layer outputs differ: diff={diff}"
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# TC-08: LayerNorm 异常形状测试
def test_layernorm_exception_shapes():
    """Test LayerNorm with invalid shapes and parameters"""
    # Test 1: normalized_shape doesn't match input shape
    normalized_shape = [8, 8]
    eps = 1e-5
    elementwise_affine = True
    
    # Create LayerNorm layer
    ln = LayerNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine
    )
    
    # Test with input that has wrong last dimensions
    # LayerNorm expects last 2 dimensions to be 8x8, but we give 4x4
    wrong_shape_input = torch.randn(2, 4, 4, 4)
    
    # This should raise a RuntimeError during forward pass
    # LayerNorm checks that input shape ends with normalized_shape
    with pytest.raises(RuntimeError) as exc_info:
        ln(wrong_shape_input)
    
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in 
              ['shape', 'size', 'dimension', 'normalized', 'expected']), \
        f"Error message doesn't mention shape mismatch: {error_msg}"
    
    # Test 2: Input with too few dimensions
    # If normalized_shape has 2 elements, input needs at least 3 dimensions
    input_2d = torch.randn(8, 8)  # Only 2 dimensions
    
    with pytest.raises(RuntimeError) as exc_info:
        ln(input_2d)
    
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in 
              ['dimension', 'shape', 'size', 'expected']), \
        f"Error message doesn't mention dimension issue: {error_msg}"
    
    # Test 3: Valid input should work
    valid_input = torch.randn(2, 4, 8, 8)  # Last 2 dimensions are 8x8
    valid_output = ln(valid_input)
    assert valid_output.shape == valid_input.shape
    
    # Test 4: Empty normalized_shape (should work for 0-dimensional normalization?)
    # Actually, normalized_shape should not be empty
    with pytest.raises(ValueError) as exc_info:
        LayerNorm(normalized_shape=[], eps=eps, elementwise_affine=elementwise_affine)
    
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in 
              ['empty', 'zero', 'size', 'dimension']), \
        f"Error message doesn't mention empty shape: {error_msg}"
    
    # Test 5: Negative values in normalized_shape
    with pytest.raises(RuntimeError) as exc_info:
        LayerNorm(normalized_shape=[-1, 8], eps=eps, elementwise_affine=elementwise_affine)
    
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in 
              ['negative', 'size', 'dimension', 'invalid']), \
        f"Error message doesn't mention negative size: {error_msg}"
    
    # Test 6: Zero in normalized_shape
    with pytest.raises(RuntimeError) as exc_info:
        LayerNorm(normalized_shape=[0, 8], eps=eps, elementwise_affine=elementwise_affine)
    
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in 
              ['zero', 'size', 'dimension', 'invalid']), \
        f"Error message doesn't mention zero size: {error_msg}"
    
    # Test 7: Very large normalized_shape
    # This should work as long as it fits in memory
    large_norm_shape = [100, 100]
    ln_large = LayerNorm(
        normalized_shape=large_norm_shape,
        eps=eps,
        elementwise_affine=elementwise_affine
    )
    
    # Create input with matching shape
    large_input = torch.randn(2, 4, 100, 100)
    large_output = ln_large(large_input)
    assert large_output.shape == large_input.shape
    
    # Test 8: normalized_shape as integer (1D normalization)
    ln_1d = LayerNorm(
        normalized_shape=16,
        eps=eps,
        elementwise_affine=elementwise_affine
    )
    
    # Test various valid 1D inputs
    test_cases_1d = [
        (2, 4, 16),      # 3D input
        (1, 16),         # 2D input
        (4, 8, 16),      # 3D with different batch/channel
    ]
    
    for shape in test_cases_1d:
        input_1d = torch.randn(*shape)
        output_1d = ln_1d(input_1d)
        assert output_1d.shape == input_1d.shape
    
    # Test 9: normalized_shape as torch.Size
    # torch is already imported at the top of the file, so no need to import again
    torch_size_shape = torch.Size([8, 8])
    ln_torchsize = LayerNorm(
        normalized_shape=torch_size_shape,
        eps=eps,
        elementwise_affine=elementwise_affine
    )
    
    torchsize_input = torch.randn(2, 4, 8, 8)
    torchsize_output = ln_torchsize(torchsize_input)
    assert torchsize_output.shape == torchsize_input.shape
    
    # Test 10: Test with scalar input (should fail)
    scalar_input = torch.tensor(5.0)
    
    with pytest.raises(RuntimeError) as exc_info:
        ln_1d(scalar_input)
    
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in 
              ['dimension', 'shape', 'size', 'expected']), \
        f"Error message doesn't mention dimension issue for scalar: {error_msg}"
    
    # Test 11: Test eps boundary values
    # eps should be positive
    with pytest.raises(ValueError) as exc_info:
        LayerNorm(normalized_shape=[8, 8], eps=-1e-5, elementwise_affine=elementwise_affine)
    
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in 
              ['eps', 'positive', 'negative', 'zero']), \
        f"Error message doesn't mention eps issue: {error_msg}"
    
    # eps = 0 might be allowed? Let's test
    try:
        ln_zero_eps = LayerNorm(normalized_shape=[8, 8], eps=0.0, elementwise_affine=elementwise_affine)
        # If it works, test forward pass
        test_input = torch.randn(2, 4, 8, 8)
        test_output = ln_zero_eps(test_input)
        assert test_output.shape == test_input.shape
    except Exception as e:
        # eps=0 might cause division by zero, which is acceptable
        pass
    
    # Test 12: Test with very small eps
    tiny_eps = 1e-10
    ln_tiny_eps = LayerNorm(
        normalized_shape=[8, 8],
        eps=tiny_eps,
        elementwise_affine=elementwise_affine
    )
    
    tiny_input = torch.randn(2, 4, 8, 8)
    tiny_output = ln_tiny_eps(tiny_input)
    assert tiny_output.shape == tiny_input.shape
    assert torch.all(torch.isfinite(tiny_output)), "Tiny eps produced non-finite values"
    
    # Test 13: Test device mismatch
    if torch.cuda.is_available():
        # Create layer on CPU
        ln_cpu = LayerNorm(normalized_shape=[8, 8], eps=eps, elementwise_affine=elementwise_affine)
        
        # Create input on CUDA
        cuda_input = torch.randn(2, 4, 8, 8).cuda()
        
        # This should work (PyTorch will move input to CPU or vice versa)
        # But might be inefficient
        try:
            cuda_output = ln_cpu(cuda_input)
            assert cuda_output.shape == cuda_input.shape
            assert cuda_output.device.type == "cpu"  # Output on CPU
        except Exception as e:
            # Device mismatch might cause issues, but that's implementation dependent
            pass
    
    # Test 14: Test dtype mismatch
    ln_float32 = LayerNorm(normalized_shape=[8, 8], eps=eps, elementwise_affine=elementwise_affine)
    
    # Create float64 input
    float64_input = torch.randn(2, 4, 8, 8).double()
    
    # This should work (PyTorch will cast)
    float64_output = ln_float32(float64_input)
    assert float64_output.shape == float64_input.shape
    assert float64_output.dtype == torch.float32  # Output in layer's dtype
    
    # Test 15: Test with requires_grad and invalid shapes
    invalid_input = torch.randn(2, 4, 4, 4, requires_grad=True)
    
    with pytest.raises(RuntimeError) as exc_info:
        ln(invalid_input)
    
    # Gradient should not be computed for invalid input
    assert invalid_input.grad is None, "Gradient should not exist for failed forward pass"
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Additional test classes and helper functions can be added here
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====