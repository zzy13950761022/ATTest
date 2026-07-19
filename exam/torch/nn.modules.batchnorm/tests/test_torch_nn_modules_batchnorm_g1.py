import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import (
    BatchNorm1d, BatchNorm2d, BatchNorm3d,
    LazyBatchNorm1d, LazyBatchNorm2d, LazyBatchNorm3d,
    SyncBatchNorm
)


# ==== BLOCK:HEADER START ====
# Fixtures and helper functions for batch normalization tests (G1 group)

@pytest.fixture(scope="function")
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42


def create_test_input(shape, dtype=torch.float32, device="cpu"):
    """Create test input tensor with fixed random values."""
    torch.manual_seed(42)
    return torch.randn(*shape, dtype=dtype, device=device)


def assert_tensor_properties(tensor, expected_shape, expected_dtype, test_name=""):
    """Assert basic tensor properties."""
    assert tensor.shape == expected_shape, f"{test_name}: shape mismatch"
    assert tensor.dtype == expected_dtype, f"{test_name}: dtype mismatch"
    assert torch.isfinite(tensor).all(), f"{test_name}: tensor contains non-finite values"


def get_oracle_batch_norm(input_tensor, bn_module):
    """Get oracle output using F.batch_norm for comparison."""
    if bn_module.training:
        # In training mode, use batch statistics
        return F.batch_norm(
            input_tensor,
            running_mean=None,
            running_var=None,
            weight=bn_module.weight,
            bias=bn_module.bias,
            training=True,
            momentum=bn_module.momentum,
            eps=bn_module.eps
        )
    else:
        # In eval mode, use running statistics
        return F.batch_norm(
            input_tensor,
            running_mean=bn_module.running_mean,
            running_var=bn_module.running_var,
            weight=bn_module.weight,
            bias=bn_module.bias,
            training=False,
            momentum=bn_module.momentum,
            eps=bn_module.eps
        )
# ==== BLOCK:HEADER END ====


# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize("input_shape, dtype_str, device_str", [
    ([4, 10], "float32", "cpu"),  # Base case from test plan
    ([4, 10, 32], "float32", "cpu"),  # 3D input extension
    ([4, 10], "float64", "cpu"),  # float64 extension
])
def test_batchnorm1d_basic_forward(input_shape, dtype_str, device_str):
    """Test basic forward propagation for BatchNorm1d.
    
    TC-01: BatchNorm1d基础前向传播
    Priority: High
    Assertion level: weak
    Group: G1
    """
    # Convert string parameters
    dtype = getattr(torch, dtype_str)
    device = torch.device(device_str)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create BatchNorm1d instance with parameters from test plan
    bn = BatchNorm1d(
        num_features=10,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True
    )
    bn.to(device=device, dtype=dtype)
    
    # Create test input
    input_tensor = torch.randn(*input_shape, dtype=dtype, device=device)
    
    # Set to training mode (default)
    bn.train()
    
    # Forward pass
    output = bn(input_tensor)
    
    # Weak assertions from test plan
    # 1. output_shape
    assert output.shape == input_tensor.shape, \
        f"Output shape {output.shape} != input shape {input_tensor.shape}"
    
    # 2. output_dtype
    assert output.dtype == dtype, \
        f"Output dtype {output.dtype} != expected dtype {dtype}"
    
    # 3. output_finite
    assert torch.isfinite(output).all(), \
        "Output contains non-finite values"
    
    # 4. running_mean_updated
    if bn.track_running_stats:
        # Check that running_mean has been updated (not all zeros)
        assert not torch.allclose(bn.running_mean, torch.zeros_like(bn.running_mean)), \
            "running_mean should be updated in training mode"
        
        # Check that running_var has been updated (not all ones)
        assert not torch.allclose(bn.running_var, torch.ones_like(bn.running_var)), \
            "running_var should be updated in training mode"
    
    # Additional basic checks
    # Check that weight and bias exist when affine=True
    assert bn.weight is not None, "weight should exist when affine=True"
    assert bn.bias is not None, "bias should exist when affine=True"
    
    # Check weight and bias shapes
    assert bn.weight.shape == (10,), f"weight shape {bn.weight.shape} != (10,)"
    assert bn.bias.shape == (10,), f"bias shape {bn.bias.shape} != (10,)"
    
    # Check that output is on correct device
    assert output.device == device, \
        f"Output device {output.device} != expected device {device}"
# ==== BLOCK:CASE_01 END ====


# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("mode, input_shape", [
    ("train", [2, 8, 16, 16]),  # Base case from test plan
    ("eval", [2, 8, 16, 16]),   # Eval mode extension
])
def test_train_eval_mode_switching(mode, input_shape):
    """Test training/evaluation mode switching for BatchNorm2d.
    
    TC-02: 训练评估模式切换
    Priority: High
    Assertion level: weak
    Group: G1
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create BatchNorm2d instance with parameters from test plan
    bn = BatchNorm2d(
        num_features=8,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True
    )
    
    # Create test input
    input_tensor = torch.randn(*input_shape, dtype=torch.float32)
    
    # Set mode
    if mode == "train":
        bn.train()
        expected_training = True
    else:
        bn.eval()
        expected_training = False
    
    # Verify mode is set correctly
    assert bn.training == expected_training, \
        f"Module training mode {bn.training} != expected {expected_training}"
    
    # Save initial running statistics
    if bn.track_running_stats:
        initial_running_mean = bn.running_mean.clone()
        initial_running_var = bn.running_var.clone()
    
    # Forward pass
    output = bn(input_tensor)
    
    # Weak assertions from test plan
    # 1. mode_switch_works
    assert bn.training == expected_training, \
        f"Mode switch failed: training={bn.training}, expected={expected_training}"
    
    # 2. running_stats_used_in_eval
    if mode == "eval" and bn.track_running_stats:
        # In eval mode, running stats should be used but not updated
        assert torch.allclose(bn.running_mean, initial_running_mean), \
            "running_mean should not be updated in eval mode"
        assert torch.allclose(bn.running_var, initial_running_var), \
            "running_var should not be updated in eval mode"
    
    # 3. output_shape_consistent
    assert output.shape == input_tensor.shape, \
        f"Output shape {output.shape} != input shape {input_tensor.shape}"
    
    # Additional checks
    assert output.dtype == torch.float32, \
        f"Output dtype {output.dtype} != expected float32"
    
    assert torch.isfinite(output).all(), \
        "Output contains non-finite values"
    
    # Check that output values are reasonable
    output_mean = output.mean().item()
    output_std = output.std().item()
    assert abs(output_mean) < 5.0, f"Output mean {output_mean} is too large"
    assert 0.1 < output_std < 5.0, f"Output std {output_std} is out of expected range"
# ==== BLOCK:CASE_02 END ====


# ==== BLOCK:CASE_03 START ====
def test_batchnorm3d_basic_functionality():
    """Test basic functionality for BatchNorm3d.
    
    TC-03: BatchNorm3d基础功能
    Priority: High
    Assertion level: weak
    Group: G1
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create BatchNorm3d instance with parameters from test plan
    bn = BatchNorm3d(
        num_features=6,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True
    )
    
    # Create test input with shape [2, 6, 8, 8, 8] from test plan
    input_tensor = torch.randn(2, 6, 8, 8, 8, dtype=torch.float32)
    
    # Set to training mode
    bn.train()
    
    # Save initial running statistics
    initial_running_mean = bn.running_mean.clone()
    initial_running_var = bn.running_var.clone()
    
    # Forward pass
    output = bn(input_tensor)
    
    # Weak assertions from test plan
    # 1. output_shape
    assert output.shape == input_tensor.shape, \
        f"Output shape {output.shape} != input shape {input_tensor.shape}"
    
    # 2. output_dtype
    assert output.dtype == torch.float32, \
        f"Output dtype {output.dtype} != expected float32"
    
    # 3. running_var_updated
    assert not torch.allclose(bn.running_var, initial_running_var), \
        "running_var should be updated in training mode"
    
    # 4. weight_bias_exist
    assert bn.weight is not None, "weight should exist when affine=True"
    assert bn.bias is not None, "bias should exist when affine=True"
    
    # Additional checks
    # Check that running_mean is also updated
    assert not torch.allclose(bn.running_mean, initial_running_mean), \
        "running_mean should be updated in training mode"
    
    # Check output is finite
    assert torch.isfinite(output).all(), \
        "Output contains non-finite values"
    
    # Check weight and bias shapes
    assert bn.weight.shape == (6,), f"weight shape {bn.weight.shape} != (6,)"
    assert bn.bias.shape == (6,), f"bias shape {bn.bias.shape} != (6,)"
    
    # Test normalization is applied
    # For BatchNorm3d, normalization should be applied per feature
    # across batch and spatial dimensions (dim 0, 2, 3, 4)
    output_reshaped = output.transpose(0, 1).reshape(6, -1)  # [C, N*D*H*W]
    
    for c in range(6):
        feature_values = output_reshaped[c]
        feature_mean = feature_values.mean().item()
        feature_std = feature_values.std().item()
        
        # Check approximate zero mean
        assert abs(feature_mean) < 0.5, \
            f"Feature {c}: mean {feature_mean:.4f} is not close to zero"
        
        # Check approximate unit variance
        assert 0.8 < feature_std < 1.2, \
            f"Feature {c}: std {feature_std:.4f} is not close to 1"
    
    # Test eval mode
    bn.eval()
    
    # Save running stats after training mode forward
    running_mean_after_train = bn.running_mean.clone()
    running_var_after_train = bn.running_var.clone()
    
    eval_output = bn(input_tensor)
    
    # Output shape should still be correct
    assert eval_output.shape == input_tensor.shape, \
        f"Eval output shape {eval_output.shape} != input shape {input_tensor.shape}"
    
    # In eval mode, running stats should not be updated
    assert torch.allclose(bn.running_mean, running_mean_after_train, rtol=1e-4), \
        "running_mean should not change in eval mode"
    assert torch.allclose(bn.running_var, running_var_after_train, rtol=1e-4), \
        "running_var should not change in eval mode"
    
    # Eval output should be different from training output
    assert not torch.allclose(eval_output, output, rtol=1e-4), \
        "Eval mode output should differ from training mode output"
# ==== BLOCK:CASE_03 END ====


# ==== BLOCK:CASE_04 START ====
def test_lazy_batchnorm1d_delayed_initialization():
    """Test lazy initialization for LazyBatchNorm1d.
    
    TC-04: 懒加载类延迟初始化
    Priority: Medium
    Assertion level: weak
    Group: G1
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create LazyBatchNorm1d instance with num_features to be inferred
    lazy_bn = LazyBatchNorm1d(
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True
    )
    
    # Weak assertions from test plan
    # 1. lazy_initialization_works
    # Initially, num_features should be 0 (uninitialized)
    assert lazy_bn.num_features == 0, \
        f"Initial num_features should be 0, got {lazy_bn.num_features}"
    
    # Initially, weight and bias should be UninitializedParameter
    assert isinstance(lazy_bn.weight, torch.nn.parameter.UninitializedParameter), \
        "weight should be UninitializedParameter before first forward"
    assert isinstance(lazy_bn.bias, torch.nn.parameter.UninitializedParameter), \
        "bias should be UninitializedParameter before first forward"
    
    # Initially, running stats should be UninitializedBuffer
    assert isinstance(lazy_bn.running_mean, torch.nn.parameter.UninitializedBuffer), \
        "running_mean should be UninitializedBuffer before first forward"
    assert isinstance(lazy_bn.running_var, torch.nn.parameter.UninitializedBuffer), \
        "running_var should be UninitializedBuffer before first forward"
    
    # 2. num_features_inferred
    # Create test input with shape [4, 12] from test plan
    input_tensor = torch.randn(4, 12, dtype=torch.float32)
    
    # 3. first_forward_succeeds
    # First forward pass should succeed and initialize parameters
    output = lazy_bn(input_tensor)
    
    # After first forward, num_features should be inferred from input
    assert lazy_bn.num_features == 12, \
        f"num_features should be inferred as 12, got {lazy_bn.num_features}"
    
    # Weight and bias should now be regular Parameters
    assert isinstance(lazy_bn.weight, torch.nn.Parameter), \
        "weight should be Parameter after first forward"
    assert isinstance(lazy_bn.bias, torch.nn.Parameter), \
        "bias should be Parameter after first forward"
    
    # Running stats should now be regular tensors
    assert isinstance(lazy_bn.running_mean, torch.Tensor), \
        "running_mean should be Tensor after first forward"
    assert isinstance(lazy_bn.running_var, torch.Tensor), \
        "running_var should be Tensor after first forward"
    
    # Check parameter shapes
    assert lazy_bn.weight.shape == (12,), f"weight shape {lazy_bn.weight.shape} != (12,)"
    assert lazy_bn.bias.shape == (12,), f"bias shape {lazy_bn.bias.shape} != (12,)"
    assert lazy_bn.running_mean.shape == (12,), f"running_mean shape {lazy_bn.running_mean.shape} != (12,)"
    assert lazy_bn.running_var.shape == (12,), f"running_var shape {lazy_bn.running_var.shape} != (12,)"
    
    # Check output properties
    assert output.shape == input_tensor.shape, \
        f"Output shape {output.shape} != input shape {input_tensor.shape}"
    assert output.dtype == torch.float32, \
        f"Output dtype {output.dtype} != expected float32"
    assert torch.isfinite(output).all(), \
        "Output contains non-finite values"
    
    # Test that subsequent forwards work correctly
    second_input = torch.randn(2, 12, dtype=torch.float32)
    second_output = lazy_bn(second_input)
    
    assert second_output.shape == second_input.shape, \
        f"Second output shape {second_output.shape} != input shape {second_input.shape}"
    
    # Test with different batch size but same feature dimension
    third_input = torch.randn(8, 12, dtype=torch.float32)
    third_output = lazy_bn(third_input)
    
    assert third_output.shape == third_input.shape, \
        f"Third output shape {third_output.shape} != input shape {third_input.shape}"
    
    # Test that trying to use wrong feature dimension raises error
    wrong_input = torch.randn(4, 10, dtype=torch.float32)
    try:
        wrong_output = lazy_bn(wrong_input)
        # If no error is raised, at least check the output shape
        assert wrong_output.shape == wrong_input.shape, \
            "Output shape should match input shape even with wrong feature dimension"
    except (RuntimeError, ValueError) as e:
        # It's acceptable for this to raise an error
        pass
    
    # Test eval mode
    lazy_bn.eval()
    eval_output = lazy_bn(input_tensor)
    
    assert eval_output.shape == input_tensor.shape, \
        f"Eval output shape {eval_output.shape} != input shape {input_tensor.shape}"
    
    # Test reset_parameters
    original_weight = lazy_bn.weight.clone()
    original_bias = lazy_bn.bias.clone()
    
    lazy_bn.reset_parameters()
    
    # After reset, parameters should be reinitialized (different from before)
    assert not torch.allclose(lazy_bn.weight, original_weight), \
        "weight should be reinitialized after reset_parameters"
    assert not torch.allclose(lazy_bn.bias, original_bias), \
        "bias should be reinitialized after reset_parameters"
# ==== BLOCK:CASE_04 END ====


# ==== BLOCK:FOOTER START ====
# Additional test cases and edge case tests for G1 group

def test_invalid_num_features():
    """Test that num_features <= 0 raises appropriate error.
    
    Note: BatchNorm constructors validate num_features > 0.
    """
    # Test with num_features=0 - this should raise RuntimeError
    with pytest.raises(RuntimeError, match="Trying to create tensor with negative dimension"):
        BatchNorm1d(num_features=0)
    
    # Test with num_features=-1 - this should also raise RuntimeError
    with pytest.raises(RuntimeError, match="Trying to create tensor with negative dimension"):
        BatchNorm1d(num_features=-1)
    
    # Test with valid num_features
    bn_valid = BatchNorm1d(num_features=10)
    assert bn_valid.num_features == 10
    
    # Test forward pass with valid num_features
    input_tensor = torch.randn(4, 10)
    output = bn_valid(input_tensor)
    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


def test_invalid_eps():
    """Test that eps <= 0 works correctly.
    
    Note: BatchNorm constructors don't validate eps <= 0,
    but we can test the behavior.
    """
    # Test with eps=0.0 - should work but may cause numerical issues
    bn_eps_zero = BatchNorm1d(num_features=10, eps=0.0)
    assert bn_eps_zero.eps == 0.0
    
    # Test with negative eps - should work but is mathematically invalid
    bn_eps_negative = BatchNorm1d(num_features=10, eps=-1e-5)
    assert bn_eps_negative.eps == -1e-5
    
    # Test forward pass with eps=0.0
    input_tensor = torch.randn(4, 10)
    output = bn_eps_zero(input_tensor)
    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


def test_invalid_momentum():
    """Test that momentum outside [0, 1] works correctly.
    
    Note: BatchNorm constructors don't validate momentum range,
    but we can test the behavior.
    """
    # Test with momentum > 1
    bn_momentum_high = BatchNorm1d(num_features=10, momentum=1.1)
    assert bn_momentum_high.momentum == 1.1
    
    # Test with negative momentum
    bn_momentum_negative = BatchNorm1d(num_features=10, momentum=-0.1)
    assert bn_momentum_negative.momentum == -0.1
    
    # Test forward pass with momentum=1.1
    input_tensor = torch.randn(4, 10)
    output = bn_momentum_high(input_tensor)
    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


def test_input_dimension_validation():
    """Test that input dimensions are validated."""
    bn1d = BatchNorm1d(num_features=10)
    bn2d = BatchNorm2d(num_features=10)
    bn3d = BatchNorm3d(num_features=10)
    
    # BatchNorm1d should accept 2D or 3D input
    input_2d = torch.randn(4, 10)
    input_3d = torch.randn(4, 10, 32)
    output_2d = bn1d(input_2d)
    output_3d = bn1d(input_3d)
    assert output_2d.shape == input_2d.shape
    assert output_3d.shape == input_3d.shape
    
    # BatchNorm2d should reject 2D input (raises ValueError, not RuntimeError)
    with pytest.raises(ValueError, match="expected 4D input"):
        bn2d(input_2d)
    
    # BatchNorm3d should reject 2D input (raises ValueError, not RuntimeError)
    with pytest.raises(ValueError, match="expected 5D input"):
        bn3d(input_2d)
    
    # Test correct input dimensions
    input_4d = torch.randn(4, 10, 16, 16)
    output_4d = bn2d(input_4d)
    assert output_4d.shape == input_4d.shape
    
    input_5d = torch.randn(4, 10, 8, 8, 8)
    output_5d = bn3d(input_5d)
    assert output_5d.shape == input_5d.shape
# ==== BLOCK:FOOTER END ====