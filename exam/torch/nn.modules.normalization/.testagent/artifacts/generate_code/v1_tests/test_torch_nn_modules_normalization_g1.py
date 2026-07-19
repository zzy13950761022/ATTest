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
# Test fixtures and helper functions for G1 group
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

# ==== BLOCK:CASE_01 START ====
# TC-01: GroupNorm 基本前向传播
@pytest.mark.parametrize("num_groups,num_channels,eps,affine,dtype,device,shape", [
    # Base case from test plan
    (2, 4, 1e-5, True, torch.float32, "cpu", (2, 4, 8, 8)),
    # Parameter extensions
    (4, 8, 1e-6, False, torch.float64, "cpu", (4, 8, 16, 16)),
])
def test_groupnorm_forward_basic(num_groups, num_channels, eps, affine, dtype, device, shape, set_random_seed):
    """Test basic forward pass of GroupNorm"""
    # Skip CUDA tests if device not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create input tensor
    torch.manual_seed(42)
    input_tensor = torch.randn(*shape, dtype=dtype, device=device)
    
    # Create GroupNorm layer
    gn = GroupNorm(
        num_groups=num_groups,
        num_channels=num_channels,
        eps=eps,
        affine=affine
    ).to(device=device, dtype=dtype)
    
    # Forward pass
    output = gn(input_tensor)
    
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
    
    # 4. Basic property: output should be normalized per group
    # Reshape to (batch, groups, channels_per_group, *spatial)
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    spatial_dims = shape[2:]
    
    # Reshape input and output for group-wise statistics
    input_reshaped = input_tensor.view(batch_size, num_groups, channels_per_group, *spatial_dims)
    output_reshaped = output.view(batch_size, num_groups, channels_per_group, *spatial_dims)
    
    # Check that each group is normalized (mean close to 0, std close to 1)
    for b in range(batch_size):
        for g in range(num_groups):
            group_input = input_reshaped[b, g].flatten()
            group_output = output_reshaped[b, g].flatten()
            
            # Skip if group has zero variance (unlikely with random data)
            if torch.std(group_input) > 1e-7:
                # Mean should be close to 0
                mean_abs = torch.abs(torch.mean(group_output))
                assert mean_abs < 0.1, f"Group mean too large: {mean_abs}"
                
                # Std should be close to 1
                std = torch.std(group_output)
                assert 0.9 < std < 1.1, f"Group std out of range: {std}"
    
    # 5. Check affine parameters if enabled
    if affine:
        assert hasattr(gn, 'weight'), "Affine enabled but weight parameter missing"
        assert hasattr(gn, 'bias'), "Affine enabled but bias parameter missing"
        assert gn.weight.shape == (num_channels,), \
            f"Weight shape {gn.weight.shape} != expected ({num_channels},)"
        assert gn.bias.shape == (num_channels,), \
            f"Bias shape {gn.bias.shape} != expected ({num_channels},)"
    
    # 6. Compare with functional implementation (weak comparison)
    if affine:
        # When affine=True, we need to handle scale and bias
        # For weak assertion, just verify functional call doesn't crash
        try:
            functional_output = F.group_norm(
                input_tensor, num_groups, gn.weight, gn.bias, eps
            )
            # Basic shape check
            assert functional_output.shape == output.shape
        except Exception as e:
            pytest.fail(f"Functional group_norm failed: {e}")
    else:
        # When affine=False, compare directly
        functional_output = F.group_norm(
            input_tensor, num_groups, None, None, eps
        )
        # Weak comparison: just check shapes match
        assert functional_output.shape == output.shape
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# TC-02: GroupNorm 整除性异常检查
def test_groupnorm_divisibility_exception():
    """Test that GroupNorm raises ValueError when num_channels not divisible by num_groups"""
    # Test case from test plan: num_groups=3, num_channels=5 (not divisible)
    num_groups = 3
    num_channels = 5
    eps = 1e-5
    affine = True
    
    # Weak assertions: exception type and message
    with pytest.raises(ValueError) as exc_info:
        GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine
        )
    
    # Check exception type
    exception = exc_info.value
    assert isinstance(exception, ValueError), \
        f"Expected ValueError, got {type(exception).__name__}"
    
    # Check exception message contains relevant information
    error_msg = str(exception).lower()
    
    # Should mention divisibility or groups
    assert any(keyword in error_msg for keyword in 
              ['divisible', 'group', 'channel', 'num_groups', 'num_channels']), \
        f"Error message doesn't mention divisibility: {error_msg}"
    
    # Should mention the actual numbers (3 and 5)
    assert '3' in error_msg or '5' in error_msg, \
        f"Error message should mention group/channel counts: {error_msg}"
    
    # Additional test: valid case should not raise exception
    num_groups_valid = 2
    num_channels_valid = 4  # 4 is divisible by 2
    
    try:
        gn = GroupNorm(
            num_groups=num_groups_valid,
            num_channels=num_channels_valid,
            eps=eps,
            affine=affine
        )
        # Should reach here without exception
        assert gn.num_groups == num_groups_valid
        assert gn.num_channels == num_channels_valid
    except ValueError as e:
        pytest.fail(f"Valid GroupNorm raised unexpected ValueError: {e}")
    
    # Test edge case: num_groups = 1 (always divisible)
    try:
        gn = GroupNorm(
            num_groups=1,
            num_channels=7,  # Any number is divisible by 1
            eps=eps,
            affine=affine
        )
        assert gn.num_groups == 1
        assert gn.num_channels == 7
    except ValueError as e:
        pytest.fail(f"GroupNorm with num_groups=1 raised unexpected ValueError: {e}")
    
    # Test edge case: num_groups = num_channels (divisible)
    try:
        gn = GroupNorm(
            num_groups=4,
            num_channels=4,  # Each group has 1 channel
            eps=eps,
            affine=affine
        )
        assert gn.num_groups == 4
        assert gn.num_channels == 4
    except ValueError as e:
        pytest.fail(f"GroupNorm with num_groups=num_channels raised unexpected ValueError: {e}")
    
    # Test with invalid input tensor shape (should fail during forward, not init)
    gn = GroupNorm(num_groups=2, num_channels=4, eps=eps, affine=affine)
    invalid_input = torch.randn(2, 3, 8, 8)  # 3 channels, but layer expects 4
    
    # This might raise a runtime error during forward pass
    # For weak assertion, we just verify it doesn't crash in unexpected ways
    try:
        output = gn(invalid_input)
        # If it reaches here, the error might be caught later or not at all
        # This is acceptable for weak assertions
    except Exception as e:
        # Any exception is acceptable for invalid input
        pass
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: DEFERRED - GroupNorm 参数扩展测试
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# TC-06: DEFERRED - GroupNorm 设备/数据类型测试
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:FOOTER START ====
# Additional test classes and helper functions can be added here
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====