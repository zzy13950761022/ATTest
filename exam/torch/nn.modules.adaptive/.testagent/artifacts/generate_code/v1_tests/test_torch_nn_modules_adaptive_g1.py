import math
import pytest
import torch
import torch.nn as nn
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G1 group

import numpy as np
import random
from typing import Tuple, List

# Set random seeds for reproducibility
@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility across all tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    yield

def create_test_data(
    in_features: int,
    n_classes: int,
    batch_size: int = 2,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create test input and target tensors.
    
    Args:
        in_features: Input feature dimension
        n_classes: Number of classes
        batch_size: Batch size (0 for non-batched input)
        dtype: Data type
        device: Device to place tensors on
        
    Returns:
        Tuple of (input_tensor, target_tensor)
    """
    if batch_size == 0:
        # Non-batched input
        input_tensor = torch.randn(in_features, dtype=dtype, device=device)
        target_tensor = torch.randint(0, n_classes, (), dtype=torch.long, device=device)
    else:
        # Batched input
        input_tensor = torch.randn(batch_size, in_features, dtype=dtype, device=device)
        target_tensor = torch.randint(0, n_classes, (batch_size,), dtype=torch.long, device=device)
    
    return input_tensor, target_tensor

def create_adaptive_softmax(
    in_features: int,
    n_classes: int,
    cutoffs: List[int],
    div_value: float = 4.0,
    head_bias: bool = False,
    device: str = "cpu"
) -> AdaptiveLogSoftmaxWithLoss:
    """
    Create an AdaptiveLogSoftmaxWithLoss instance.
    
    Args:
        in_features: Input feature dimension
        n_classes: Number of classes
        cutoffs: Cutoff values for clustering
        div_value: Division value for cluster size computation
        head_bias: Whether to add bias to the head
        device: Device to place module on
        
    Returns:
        AdaptiveLogSoftmaxWithLoss instance
    """
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=in_features,
        n_classes=n_classes,
        cutoffs=cutoffs,
        div_value=div_value,
        head_bias=head_bias
    )
    return model.to(device)
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# Test case: 基本前向传播功能

@pytest.mark.parametrize(
    "in_features,n_classes,cutoffs,div_value,head_bias,batch_size,dtype,device",
    [
        # Base case from test plan
        (10, 100, [10, 50], 4.0, False, 2, torch.float32, "cpu"),
        # Parameter extension
        (20, 200, [20, 100], 2.0, True, 4, torch.float64, "cpu"),
    ]
)
def test_adaptive_softmax_forward_basic(
    in_features: int,
    n_classes: int,
    cutoffs: List[int],
    div_value: float,
    head_bias: bool,
    batch_size: int,
    dtype: torch.dtype,
    device: str
):
    """
    Test basic forward propagation functionality.
    
    Weak assertions:
    1. output_shape: Output tensor has correct shape
    2. loss_scalar: Loss is a scalar value
    3. finite_values: All output values are finite
    4. dtype_match: Output has correct data type
    """
    # Create test data
    input_tensor, target_tensor = create_test_data(
        in_features=in_features,
        n_classes=n_classes,
        batch_size=batch_size,
        dtype=dtype,
        device=device
    )
    
    # Create model
    model = create_adaptive_softmax(
        in_features=in_features,
        n_classes=n_classes,
        cutoffs=cutoffs,
        div_value=div_value,
        head_bias=head_bias,
        device=device
    )
    
    # Forward pass
    result = model(input_tensor, target_tensor)
    
    # Assertion 1: output_shape
    if batch_size == 0:
        # Non-batched input should produce scalar output
        assert result.output.shape == (), f"Expected scalar output for non-batched input, got {result.output.shape}"
    else:
        # Batched input should produce (batch_size,) output
        assert result.output.shape == (batch_size,), f"Expected output shape ({batch_size},), got {result.output.shape}"
    
    # Assertion 2: loss_scalar
    assert result.loss.shape == (), f"Expected scalar loss, got {result.loss.shape}"
    
    # Assertion 3: finite_values
    assert torch.all(torch.isfinite(result.output)), "Output contains non-finite values"
    assert torch.isfinite(result.loss), "Loss is not finite"
    
    # Assertion 4: dtype_match
    assert result.output.dtype == dtype, f"Expected output dtype {dtype}, got {result.output.dtype}"
    assert result.loss.dtype == dtype, f"Expected loss dtype {dtype}, got {result.loss.dtype}"
    
    # Additional weak assertion: loss is non-negative
    assert result.loss >= 0, f"Loss should be non-negative, got {result.loss.item()}"
    
    # Additional weak assertion: output values are reasonable (log probabilities should be <= 0)
    # Note: log probabilities can be slightly positive due to numerical issues
    assert torch.all(result.output <= 1e-5), f"Log probabilities should be <= 0, got max value {result.output.max().item()}"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# Test case: 批处理与非批处理输入

@pytest.mark.parametrize(
    "in_features,n_classes,cutoffs,div_value,head_bias,batch_size,dtype,device",
    [
        # Non-batched input
        (8, 50, [5, 20], 2.0, True, 0, torch.float32, "cpu"),
        # Batched input
        (8, 50, [5, 20], 2.0, True, 3, torch.float32, "cpu"),
    ]
)
def test_adaptive_softmax_batch_handling(
    in_features: int,
    n_classes: int,
    cutoffs: List[int],
    div_value: float,
    head_bias: bool,
    batch_size: int,
    dtype: torch.dtype,
    device: str
):
    """
    Test batch and non-batch input compatibility.
    
    Weak assertions:
    1. shape_compatibility: Output shape matches input shape
    2. no_nan: No NaN values in output
    3. device_consistency: Output is on correct device
    """
    # Create test data
    input_tensor, target_tensor = create_test_data(
        in_features=in_features,
        n_classes=n_classes,
        batch_size=batch_size,
        dtype=dtype,
        device=device
    )
    
    # Create model
    model = create_adaptive_softmax(
        in_features=in_features,
        n_classes=n_classes,
        cutoffs=cutoffs,
        div_value=div_value,
        head_bias=head_bias,
        device=device
    )
    
    # Forward pass
    result = model(input_tensor, target_tensor)
    
    # Assertion 1: shape_compatibility
    if batch_size == 0:
        # Non-batched input
        assert result.output.shape == (), f"Expected scalar output for non-batched input, got {result.output.shape}"
        assert target_tensor.shape == (), f"Expected scalar target for non-batched input, got {target_tensor.shape}"
    else:
        # Batched input
        assert result.output.shape == (batch_size,), f"Expected output shape ({batch_size},), got {result.output.shape}"
        assert target_tensor.shape == (batch_size,), f"Expected target shape ({batch_size},), got {target_tensor.shape}"
    
    # Assertion 2: no_nan
    assert not torch.any(torch.isnan(result.output)), "Output contains NaN values"
    assert not torch.isnan(result.loss), "Loss is NaN"
    
    # Assertion 3: device_consistency
    assert result.output.device == torch.device(device), f"Output device mismatch: expected {device}, got {result.output.device}"
    assert result.loss.device == torch.device(device), f"Loss device mismatch: expected {device}, got {result.loss.device}"
    
    # Additional weak assertion: loss is scalar
    assert result.loss.dim() == 0, f"Loss should be scalar, got shape {result.loss.shape}"
    
    # Additional weak assertion: output values are reasonable
    # Log probabilities should be negative or very close to 0
    assert torch.all(result.output <= 1e-5), f"Log probabilities should be <= 0, got max value {result.output.max().item()}"
    
    # Test that model can handle both batch and non-batch inputs consistently
    # by checking that the loss computation is valid
    assert torch.isfinite(result.loss), f"Loss is not finite: {result.loss.item()}"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_05 START ====
# Test case: 不同设备兼容性 (deferred)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# Test case: deferred placeholder
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:FOOTER START ====
# Additional test utilities and cleanup for G1 group

def test_adaptive_softmax_edge_cases_g1():
    """Test edge cases for AdaptiveLogSoftmaxWithLoss in G1 group."""
    # Test with minimal valid parameters
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=1,
        n_classes=2,
        cutoffs=[1],
        div_value=2.0,
        head_bias=False
    )
    
    assert model is not None, "Should create model with minimal valid parameters"
    assert model.n_classes == 2, f"Expected n_classes=2, got {model.n_classes}"
    assert model.in_features == 1, f"Expected in_features=1, got {model.in_features}"
    
    # Test forward with minimal parameters
    input_tensor = torch.randn(2, 1)
    target_tensor = torch.tensor([0, 1], dtype=torch.long)
    result = model(input_tensor, target_tensor)
    
    assert result.output.shape == (2,), f"Expected output shape (2,), got {result.output.shape}"
    assert result.loss.shape == (), f"Expected scalar loss, got {result.loss.shape}"
    
    # Test with larger parameters
    model2 = AdaptiveLogSoftmaxWithLoss(
        in_features=100,
        n_classes=1000,
        cutoffs=[100, 500],
        div_value=4.0,
        head_bias=True
    )
    
    assert model2 is not None, "Should create model with larger parameters"
    assert model2.n_classes == 1000, f"Expected n_classes=1000, got {model2.n_classes}"
    assert model2.in_features == 100, f"Expected in_features=100, got {model2.in_features}"
    
    # Test forward with larger parameters
    input_tensor2 = torch.randn(5, 100)
    target_tensor2 = torch.randint(0, 1000, (5,), dtype=torch.long)
    result2 = model2(input_tensor2, target_tensor2)
    
    assert result2.output.shape == (5,), f"Expected output shape (5,), got {result2.output.shape}"
    assert result2.loss.shape == (), f"Expected scalar loss, got {result2.loss.shape}"

def test_adaptive_softmax_shape_mismatch():
    """Test shape mismatch error handling."""
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=10,
        n_classes=100,
        cutoffs=[10, 50],
        div_value=4.0,
        head_bias=False
    )
    
    # Test input shape mismatch (wrong in_features)
    input_tensor = torch.randn(3, 8)  # Wrong: 8 instead of 10
    target_tensor = torch.randint(0, 100, (3,), dtype=torch.long)
    
    with pytest.raises(RuntimeError) as exc_info:
        model(input_tensor, target_tensor)
    
    # Test target shape mismatch (wrong batch size)
    input_tensor2 = torch.randn(3, 10)  # Correct shape
    target_tensor2 = torch.randint(0, 100, (4,), dtype=torch.long)  # Wrong: 4 instead of 3
    
    with pytest.raises(RuntimeError) as exc_info2:
        model(input_tensor2, target_tensor2)
    
    # Test target out of range
    input_tensor3 = torch.randn(2, 10)
    target_tensor3 = torch.tensor([-1, 100], dtype=torch.long)  # Invalid: -1 and 100 (n_classes=100)
    
    with pytest.raises(IndexError) as exc_info3:
        model(input_tensor3, target_tensor3)
# ==== BLOCK:FOOTER END ====