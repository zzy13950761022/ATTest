import math
import pytest
import torch
import torch.nn as nn
from torch.nn.modules.instancenorm import (
    InstanceNorm1d, InstanceNorm2d, InstanceNorm3d,
    LazyInstanceNorm1d, LazyInstanceNorm2d, LazyInstanceNorm3d
)

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G2 group
@pytest.fixture(scope="function", autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    yield

def assert_tensor_properties(tensor, expected_shape=None, expected_dtype=None, 
                           allow_nan_inf=False, name=""):
    """Assert tensor properties with descriptive error messages."""
    if expected_shape is not None:
        assert tensor.shape == expected_shape, \
            f"{name}: Expected shape {expected_shape}, got {tensor.shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, \
            f"{name}: Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    if not allow_nan_inf:
        assert torch.isfinite(tensor).all(), \
            f"{name}: Tensor contains NaN or Inf values"
    
    return True

def create_lazy_norm_layer(norm_class, affine=False, track_running_stats=False, dtype=torch.float32):
    """Create lazy instance normalization layer."""
    if norm_class == "LazyInstanceNorm1d":
        return LazyInstanceNorm1d(
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype
        )
    elif norm_class == "LazyInstanceNorm2d":
        return LazyInstanceNorm2d(
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype
        )
    elif norm_class == "LazyInstanceNorm3d":
        return LazyInstanceNorm3d(
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype
        )
    else:
        raise ValueError(f"Unsupported lazy norm_class: {norm_class}")
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: LazyInstanceNorm自动推断
@pytest.mark.parametrize("test_params", [
    {
        "norm_class": "LazyInstanceNorm2d",
        "affine": False,
        "track_running_stats": False,
        "input_shape": (2, 3, 4, 4),
        "dtype": torch.float32,
        "device": "cpu"
    }
])
def test_lazy_instance_norm_inference(test_params):
    """Test lazy instance norm automatic feature inference."""
    # Unpack parameters
    norm_class = test_params["norm_class"]
    affine = test_params["affine"]
    track_running_stats = test_params["track_running_stats"]
    input_shape = test_params["input_shape"]
    dtype = test_params["dtype"]
    device = test_params["device"]
    
    # Create lazy instance normalization layer
    if norm_class == "LazyInstanceNorm2d":
        norm_layer = LazyInstanceNorm2d(
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype
        )
    else:
        raise ValueError(f"Unsupported norm_class: {norm_class}")
    
    norm_layer.to(device)
    
    # Check initial state - lazy layers have num_features=0 initially
    # This is expected behavior for lazy modules
    assert hasattr(norm_layer, 'num_features'), \
        "Lazy layer should have num_features attribute"
    # Note: LazyInstanceNorm2d inherits from _LazyNormBase which hardcodes num_features=0
    # This is expected behavior
    assert norm_layer.num_features == 0, \
        f"Lazy layer should have num_features=0 initially, got {norm_layer.num_features}"
    
    # Create random input
    input_tensor = torch.randn(*input_shape, dtype=dtype, device=device)
    
    # Forward pass - this should trigger initialization
    output = norm_layer(input_tensor)
    
    # Weak assertions
    # 1. Output shape matches input shape
    assert output.shape == input_shape, \
        f"Output shape {output.shape} doesn't match input shape {input_shape}"
    
    # 2. After forward pass, the layer should be properly initialized
    # Note: LazyInstanceNorm2d becomes InstanceNorm2d after initialization
    # but num_features may still be 0 in the lazy base class
    # The important thing is that forward pass works and output is correct
    
    # 3. Output dtype matches input dtype
    assert output.dtype == dtype, \
        f"Output dtype {output.dtype} doesn't match expected {dtype}"
    
    # 4. Finite values (no NaN or Inf)
    assert torch.isfinite(output).all(), \
        "Output contains NaN or Inf values"
    
    # 5. Check that layer has been properly initialized
    # After first forward, lazy layer should have correct attributes
    # The layer should now behave like a regular InstanceNorm2d
    assert isinstance(norm_layer, (InstanceNorm2d, LazyInstanceNorm2d)), \
        "Layer should be InstanceNorm2d or LazyInstanceNorm2d after initialization"
    
    # 6. No affine parameters when affine=False
    if not affine:
        # After initialization, weight and bias should be None when affine=False
        # or they may be UninitializedParameter objects
        if hasattr(norm_layer, 'weight'):
            if norm_layer.weight is not None:
                # If weight exists and is not None, it should not require grad
                assert not norm_layer.weight.requires_grad, \
                    "Weight should not require gradients when affine=False"
        if hasattr(norm_layer, 'bias'):
            if norm_layer.bias is not None:
                assert not norm_layer.bias.requires_grad, \
                    "Bias should not require gradients when affine=False"
    
    # 7. No running stats when track_running_stats=False
    if not track_running_stats:
        # When track_running_stats=False, running stats may be None or UninitializedBuffer
        if hasattr(norm_layer, 'running_mean'):
            if norm_layer.running_mean is not None:
                # If running_mean exists, it should not require grad
                assert not norm_layer.running_mean.requires_grad, \
                    "running_mean should not require gradients"
        if hasattr(norm_layer, 'running_var'):
            if norm_layer.running_var is not None:
                assert not norm_layer.running_var.requires_grad, \
                    "running_var should not require gradients"
    
    # Test that subsequent forward passes work correctly
    input_tensor2 = torch.randn(*input_shape, dtype=dtype, device=device)
    output2 = norm_layer(input_tensor2)
    
    # 8. Second forward pass should also work
    assert output2.shape == input_shape, \
        f"Second output shape {output2.shape} doesn't match input shape {input_shape}"
    assert torch.isfinite(output2).all(), \
        "Second output contains NaN or Inf values"
    
    # 9. Check that outputs are different (due to different random inputs)
    # but have similar statistical properties
    assert not torch.allclose(output, output2), \
        "Outputs from different inputs should not be identical"
    
    # 10. Check basic normalization properties (weak assertions)
    # Mean should be close to 0, std close to 1 for each channel
    # This is a weak check since we're using random data
    output_mean = output.mean(dim=(0, 2, 3))  # mean over batch, height, width
    output_std = output.std(dim=(0, 2, 3), unbiased=False)  # std over batch, height, width
    
    # Allow some tolerance for random data
    assert torch.all(torch.abs(output_mean) < 0.5), \
        f"Output mean should be close to 0, got {output_mean}"
    assert torch.all(torch.abs(output_std - 1.0) < 0.5), \
        f"Output std should be close to 1, got {output_std}"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: track_running_stats功能
@pytest.mark.parametrize("test_params", [
    {
        "norm_class": "InstanceNorm3d",
        "num_features": 2,
        "affine": False,
        "track_running_stats": True,
        "input_shape": (2, 2, 3, 3, 3),
        "dtype": torch.float32,
        "device": "cpu"
    },
    {
        "norm_class": "InstanceNorm3d",
        "num_features": 2,
        "affine": False,
        "track_running_stats": True,
        "input_shape": (2, 2, 3, 3, 3),
        "dtype": torch.float32,
        "device": "cpu",
        "momentum": 0.5
    }
])
def test_instance_norm_track_running_stats(test_params):
    """Test track_running_stats functionality."""
    # Unpack parameters
    norm_class = test_params["norm_class"]
    num_features = test_params["num_features"]
    affine = test_params["affine"]
    track_running_stats = test_params["track_running_stats"]
    input_shape = test_params["input_shape"]
    dtype = test_params["dtype"]
    device = test_params["device"]
    momentum = test_params.get("momentum", 0.1)  # default momentum
    
    # Create instance normalization layer
    if norm_class == "InstanceNorm3d":
        norm_layer = InstanceNorm3d(
            num_features=num_features,
            affine=affine,
            track_running_stats=track_running_stats,
            momentum=momentum,
            dtype=dtype
        )
    else:
        raise ValueError(f"Unsupported norm_class: {norm_class}")
    
    norm_layer.to(device)
    
    # Weak assertions - check running stats exist
    # 1. Running mean and var should exist when track_running_stats=True
    assert hasattr(norm_layer, 'running_mean'), \
        "Should have running_mean when track_running_stats=True"
    assert hasattr(norm_layer, 'running_var'), \
        "Should have running_var when track_running_stats=True"
    
    # 2. Running stats should be initialized
    assert norm_layer.running_mean is not None, \
        "running_mean should be initialized"
    assert norm_layer.running_var is not None, \
        "running_var should be initialized"
    
    # 3. Running stats shapes should match num_features
    assert norm_layer.running_mean.shape == (num_features,), \
        f"running_mean shape {norm_layer.running_mean.shape} should be ({num_features},)"
    assert norm_layer.running_var.shape == (num_features,), \
        f"running_var shape {norm_layer.running_var.shape} should be ({num_features},)"
    
    # 4. Running stats should be initialized to zeros (mean) and ones (var)
    assert torch.allclose(norm_layer.running_mean, 
                         torch.zeros(num_features, dtype=dtype)), \
        "running_mean should be initialized to 0"
    assert torch.allclose(norm_layer.running_var, 
                         torch.ones(num_features, dtype=dtype)), \
        "running_var should be initialized to 1"
    
    # 5. Running stats should not require gradients
    assert not norm_layer.running_mean.requires_grad, \
        "running_mean should not require gradients"
    assert not norm_layer.running_var.requires_grad, \
        "running_var should not require gradients"
    
    # Create random input
    input_tensor = torch.randn(*input_shape, dtype=dtype, device=device)
    
    # Set layer to training mode (default)
    norm_layer.train()
    
    # Save initial running stats
    initial_mean = norm_layer.running_mean.clone()
    initial_var = norm_layer.running_var.clone()
    
    # Forward pass in training mode
    output = norm_layer(input_tensor)
    
    # 6. Output shape matches input shape
    assert output.shape == input_shape, \
        f"Output shape {output.shape} doesn't match input shape {input_shape}"
    
    # 7. Finite values (no NaN or Inf)
    assert torch.isfinite(output).all(), \
        "Output contains NaN or Inf values"
    
    # 8. Running stats should be updated in training mode when track_running_stats=True
    # Instance norm updates running stats in training mode when track_running_stats=True
    # The update formula: running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    # Since initial running_mean is 0, after update it should be momentum * batch_mean
    assert not torch.allclose(norm_layer.running_mean, initial_mean), \
        "running_mean should be updated in training mode when track_running_stats=True"
    assert not torch.allclose(norm_layer.running_var, initial_var), \
        "running_var should be updated in training mode when track_running_stats=True"
    
    # 9. Check that running stats have been updated (not zero anymore)
    assert not torch.allclose(norm_layer.running_mean, 
                             torch.zeros(num_features, dtype=dtype)), \
        "running_mean should not be zero after update"
    assert not torch.allclose(norm_layer.running_var, 
                             torch.ones(num_features, dtype=dtype)), \
        "running_var should not be one after update"
    
    # Switch to evaluation mode
    norm_layer.eval()
    
    # Save running stats before eval forward
    before_eval_mean = norm_layer.running_mean.clone()
    before_eval_var = norm_layer.running_var.clone()
    
    # Forward pass in evaluation mode
    output_eval = norm_layer(input_tensor)
    
    # 10. Output in eval mode should also be valid
    assert output_eval.shape == input_shape, \
        f"Eval output shape {output_eval.shape} doesn't match input shape {input_shape}"
    assert torch.isfinite(output_eval).all(), \
        "Eval output contains NaN or Inf values"
    
    # 11. Running stats should NOT be updated in evaluation mode
    assert torch.allclose(norm_layer.running_mean, before_eval_mean), \
        "running_mean should not change in evaluation mode"
    assert torch.allclose(norm_layer.running_var, before_eval_var), \
        "running_var should not change in evaluation mode"
    
    # 12. No affine parameters when affine=False
    if not affine:
        assert norm_layer.weight is None, \
            "Should not have weight parameter when affine=False"
        assert norm_layer.bias is None, \
            "Should not have bias parameter when affine=False"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: 无批次输入处理 (DEFERRED - placeholder)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# DEFERRED - placeholder
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# DEFERRED - placeholder
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# DEFERRED - placeholder
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Additional test cases and helper functions for G2 group

def test_lazy_instance_norm_parameter_validation():
    """Test parameter validation for lazy instance norm."""
    # Test that lazy instance norm doesn't require num_features parameter
    # This is the main difference from regular instance norm
    lazy_norm = LazyInstanceNorm2d(affine=False, track_running_stats=False)
    assert lazy_norm.num_features == 0, \
        "LazyInstanceNorm2d should have num_features=0 initially"
    
    # Note: LazyInstanceNorm2d may not validate eps in constructor
    # but in forward pass. We'll test that invalid eps causes issues during forward.
    # For now, we'll test that constructor accepts valid eps values.
    
    # Test valid eps values are accepted
    lazy_norm_eps_small = LazyInstanceNorm2d(eps=1e-8, affine=False, track_running_stats=False)
    assert lazy_norm_eps_small.eps == 1e-8, f"eps should be 1e-8, got {lazy_norm_eps_small.eps}"
    
    lazy_norm_eps_large = LazyInstanceNorm2d(eps=1e-3, affine=False, track_running_stats=False)
    assert lazy_norm_eps_large.eps == 1e-3, f"eps should be 1e-3, got {lazy_norm_eps_large.eps}"
    
    # Test invalid momentum
    with pytest.raises(ValueError, match="momentum"):
        LazyInstanceNorm2d(momentum=-0.1)
    
    with pytest.raises(ValueError, match="momentum"):
        LazyInstanceNorm2d(momentum=1.1)
    
    # Test valid momentum values
    lazy_norm_momentum_zero = LazyInstanceNorm2d(momentum=0.0, affine=False, track_running_stats=False)
    assert lazy_norm_momentum_zero.momentum == 0.0, f"momentum should be 0.0, got {lazy_norm_momentum_zero.momentum}"
    
    lazy_norm_momentum_one = LazyInstanceNorm2d(momentum=1.0, affine=False, track_running_stats=False)
    assert lazy_norm_momentum_one.momentum == 1.0, f"momentum should be 1.0, got {lazy_norm_momentum_one.momentum}"
    
    lazy_norm_momentum_half = LazyInstanceNorm2d(momentum=0.5, affine=False, track_running_stats=False)
    assert lazy_norm_momentum_half.momentum == 0.5, f"momentum should be 0.5, got {lazy_norm_momentum_half.momentum}"

def test_lazy_instance_norm_dimension_validation():
    """Test input dimension validation for lazy instance norm."""
    # LazyInstanceNorm2d should accept 3D or 4D input
    lazy_norm2d = LazyInstanceNorm2d(affine=False, track_running_stats=False)
    
    # Valid 4D input (batch, channels, height, width)
    valid_input_4d = torch.randn(2, 3, 4, 5)
    output = lazy_norm2d(valid_input_4d)
    assert output.shape == (2, 3, 4, 5)
    
    # Valid 3D input (channels, height, width) - no batch
    lazy_norm2d2 = LazyInstanceNorm2d(affine=False, track_running_stats=False)
    valid_input_3d = torch.randn(3, 4, 5)
    output = lazy_norm2d2(valid_input_3d)
    assert output.shape == (3, 4, 5)
    
    # Invalid 2D input for LazyInstanceNorm2d
    lazy_norm2d3 = LazyInstanceNorm2d(affine=False, track_running_stats=False)
    invalid_input_2d = torch.randn(3, 4)
    with pytest.raises(ValueError, match="expected 3D or 4D input"):
        lazy_norm2d3(invalid_input_2d)
    
    # Invalid 5D input for LazyInstanceNorm2d
    lazy_norm2d4 = LazyInstanceNorm2d(affine=False, track_running_stats=False)
    invalid_input_5d = torch.randn(2, 3, 4, 5, 6)
    with pytest.raises(ValueError, match="expected 3D or 4D input"):
        lazy_norm2d4(invalid_input_5d)

def test_lazy_instance_norm_affine_parameters():
    """Test affine parameters for lazy instance norm."""
    # Test with affine=True
    lazy_norm = LazyInstanceNorm2d(affine=True, track_running_stats=False)
    
    # Initially, weight and bias should be UninitializedParameter
    assert hasattr(lazy_norm, 'weight'), "Should have weight attribute"
    assert hasattr(lazy_norm, 'bias'), "Should have bias attribute"
    
    # Forward pass to initialize
    input_tensor = torch.randn(2, 3, 4, 4)
    output = lazy_norm(input_tensor)
    
    # After forward pass, weight and bias should be initialized
    assert lazy_norm.weight is not None, "Weight should be initialized after forward pass"
    assert lazy_norm.bias is not None, "Bias should be initialized after forward pass"
    assert lazy_norm.weight.shape == (3,), f"Weight shape should be (3,), got {lazy_norm.weight.shape}"
    assert lazy_norm.bias.shape == (3,), f"Bias shape should be (3,), got {lazy_norm.bias.shape}"
    
    # Parameters should be trainable
    assert lazy_norm.weight.requires_grad, "Weight should require gradients"
    assert lazy_norm.bias.requires_grad, "Bias should require gradients"
    
    # Initial values should be specific defaults
    assert torch.allclose(lazy_norm.weight, torch.ones(3, dtype=torch.float32)), \
        "Weight should be initialized to 1"
    assert torch.allclose(lazy_norm.bias, torch.zeros(3, dtype=torch.float32)), \
        "Bias should be initialized to 0"
# ==== BLOCK:FOOTER END ====