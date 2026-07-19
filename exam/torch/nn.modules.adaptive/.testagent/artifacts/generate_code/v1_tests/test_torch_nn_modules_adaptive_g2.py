import math
import pytest
import torch
import torch.nn as nn
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G2 group

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

# ==== BLOCK:CASE_03 START ====
# Test case: cutoffs 参数验证

@pytest.mark.parametrize(
    "in_features,n_classes,cutoffs,div_value,head_bias,test_type,dtype,device,should_raise",
    [
        # Valid cutoffs
        (5, 30, [5, 15], 4.0, False, "valid", torch.float32, "cpu", False),
        # Invalid: non-increasing cutoffs (parameter extension)
        (5, 30, [30, 15], 4.0, False, "invalid_non_increasing", torch.float32, "cpu", True),
        # Invalid: duplicate cutoffs (parameter extension)
        (5, 30, [5, 5, 15], 4.0, False, "invalid_duplicate", torch.float32, "cpu", True),
    ]
)
def test_adaptive_softmax_cutoffs_validation(
    in_features: int,
    n_classes: int,
    cutoffs: List[int],
    div_value: float,
    head_bias: bool,
    test_type: str,
    dtype: torch.dtype,
    device: str,
    should_raise: bool
):
    """
    Test cutoffs parameter validation.
    
    Weak assertions:
    1. no_exception: Valid cutoffs should not raise exception
    2. module_initialized: Module should be properly initialized for valid cutoffs
    """
    if should_raise:
        # Test that invalid cutoffs raise ValueError
        with pytest.raises(ValueError) as exc_info:
            model = AdaptiveLogSoftmaxWithLoss(
                in_features=in_features,
                n_classes=n_classes,
                cutoffs=cutoffs,
                div_value=div_value,
                head_bias=head_bias
            )
        
        # Verify the error message contains relevant information
        error_msg = str(exc_info.value).lower()
        if test_type == "invalid_non_increasing":
            assert "increasing" in error_msg or "sorted" in error_msg, \
                f"Expected error about non-increasing cutoffs, got: {error_msg}"
        elif test_type == "invalid_duplicate":
            assert "unique" in error_msg or "duplicate" in error_msg, \
                f"Expected error about duplicate cutoffs, got: {error_msg}"
    else:
        # Test that valid cutoffs work correctly
        try:
            model = AdaptiveLogSoftmaxWithLoss(
                in_features=in_features,
                n_classes=n_classes,
                cutoffs=cutoffs,
                div_value=div_value,
                head_bias=head_bias
            )
            
            # Assertion 1: no_exception - module created successfully
            assert model is not None, "Module should be created successfully"
            
            # Assertion 2: module_initialized
            assert hasattr(model, 'head'), "Module should have head attribute"
            assert hasattr(model, 'tail'), "Module should have tail attribute"
            assert isinstance(model.tail, nn.ModuleList), "tail should be a ModuleList"
            
            # Verify cutoffs are stored correctly
            assert model.cutoffs == cutoffs, f"Cutoffs not stored correctly: expected {cutoffs}, got {model.cutoffs}"
            
            # Verify n_classes is stored correctly
            assert model.n_classes == n_classes, f"n_classes not stored correctly: expected {n_classes}, got {model.n_classes}"
            
            # Verify in_features is stored correctly
            assert model.in_features == in_features, f"in_features not stored correctly: expected {in_features}, got {model.in_features}"
            
            # Test forward pass with small batch
            input_tensor = torch.randn(2, in_features, dtype=dtype, device=device)
            target_tensor = torch.randint(0, n_classes, (2,), dtype=torch.long, device=device)
            
            model = model.to(device)
            result = model(input_tensor, target_tensor)
            
            # Verify output shape
            assert result.output.shape == (2,), f"Expected output shape (2,), got {result.output.shape}"
            
            # Verify loss is scalar
            assert result.loss.shape == (), f"Expected scalar loss, got {result.loss.shape}"
            
            # Verify no NaN values
            assert not torch.any(torch.isnan(result.output)), "Output contains NaN values"
            assert not torch.isnan(result.loss), "Loss is NaN"
            
        except Exception as e:
            pytest.fail(f"Valid cutoffs should not raise exception: {e}")
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# Test case: log_prob 辅助方法

@pytest.mark.parametrize(
    "in_features,n_classes,cutoffs,div_value,head_bias,batch_size,dtype,device",
    [
        (12, 80, [10, 30, 60], 4.0, False, 2, torch.float32, "cpu"),
    ]
)
def test_adaptive_softmax_log_prob(
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
    Test log_prob helper method.
    
    Weak assertions:
    1. method_exists: log_prob method exists
    2. returns_tensor: Returns a tensor
    3. shape_match: Output shape matches expected shape
    """
    # Create test data
    input_tensor, _ = create_test_data(
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
    
    # Assertion 1: method_exists
    assert hasattr(model, 'log_prob'), "Model should have log_prob method"
    assert callable(model.log_prob), "log_prob should be callable"
    
    # Call log_prob method
    log_probs = model.log_prob(input_tensor)
    
    # Assertion 2: returns_tensor
    assert isinstance(log_probs, torch.Tensor), f"log_prob should return Tensor, got {type(log_probs)}"
    
    # Assertion 3: shape_match
    if batch_size == 0:
        # Non-batched input: output should be (n_classes,)
        expected_shape = (n_classes,)
    else:
        # Batched input: output should be (batch_size, n_classes)
        expected_shape = (batch_size, n_classes)
    
    assert log_probs.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {log_probs.shape}"
    
    # Additional weak assertion: no NaN values
    assert not torch.any(torch.isnan(log_probs)), "log_prob output contains NaN values"
    
    # Additional weak assertion: all values are finite
    assert torch.all(torch.isfinite(log_probs)), "log_prob output contains non-finite values"
    
    # Additional weak assertion: log probabilities should be <= 0
    # (with small tolerance for numerical errors)
    assert torch.all(log_probs <= 1e-5), \
        f"Log probabilities should be <= 0, got max value {log_probs.max().item()}"
    
    # Test consistency with forward pass for a single target
    if batch_size > 0:
        # Create target tensor
        target_tensor = torch.randint(0, n_classes, (batch_size,), dtype=torch.long, device=device)
        
        # Get forward pass result
        result = model(input_tensor, target_tensor)
        
        # For each example, the log_prob at the target index should match the output
        for i in range(batch_size):
            target_idx = target_tensor[i].item()
            log_prob_at_target = log_probs[i, target_idx]
            output_value = result.output[i]
            
            # Check they're close (allow small numerical differences)
            assert torch.allclose(log_prob_at_target, output_value, rtol=1e-5, atol=1e-8), \
                f"log_prob at target index {target_idx} doesn't match forward output: " \
                f"{log_prob_at_target.item()} vs {output_value.item()}"
    
    # Test that sum of exponentials is approximately 1 for each example
    if batch_size == 0:
        # Non-batched: sum over all classes should be ~1
        probs_sum = torch.exp(log_probs).sum().item()
        assert abs(probs_sum - 1.0) < 1e-5, \
            f"Sum of probabilities should be ~1, got {probs_sum}"
    else:
        # Batched: sum over classes for each example should be ~1
        probs_sum = torch.exp(log_probs).sum(dim=1)
        for i in range(batch_size):
            assert abs(probs_sum[i].item() - 1.0) < 1e-5, \
                f"Example {i}: sum of probabilities should be ~1, got {probs_sum[i].item()}"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_07 START ====
# Test case: deferred placeholder
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# Test case: deferred placeholder
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Additional test utilities and cleanup for G2 group

def test_adaptive_softmax_predict_method():
    """Test predict method if it exists."""
    # Create a simple model
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=10,
        n_classes=100,
        cutoffs=[10, 50],
        div_value=4.0,
        head_bias=False
    )
    
    # Check if predict method exists (it should according to documentation)
    if hasattr(model, 'predict') and callable(model.predict):
        # Test with batched input
        input_tensor = torch.randn(3, 10)
        predictions = model.predict(input_tensor)
        
        # Verify predictions are tensors
        assert isinstance(predictions, torch.Tensor), \
            f"predict should return Tensor, got {type(predictions)}"
        
        # Verify shape: (batch_size,)
        assert predictions.shape == (3,), \
            f"Expected predictions shape (3,), got {predictions.shape}"
        
        # Verify predictions are within valid class range
        assert torch.all(predictions >= 0), "Predictions should be >= 0"
        assert torch.all(predictions < 100), "Predictions should be < n_classes"
        
        # Verify predictions are integers
        assert predictions.dtype == torch.long, \
            f"Predictions should be long type, got {predictions.dtype}"

def test_adaptive_softmax_invalid_parameters_g2():
    """Test invalid parameter combinations for G2 group."""
    # Test n_classes < 2
    with pytest.raises(ValueError) as exc_info:
        AdaptiveLogSoftmaxWithLoss(
            in_features=10,
            n_classes=1,  # Invalid: n_classes must be >= 2
            cutoffs=[5],
            div_value=4.0,
            head_bias=False
        )
    assert "n_classes" in str(exc_info.value).lower(), \
        f"Expected error about n_classes, got: {exc_info.value}"
    
    # Test in_features <= 0
    with pytest.raises(ValueError) as exc_info:
        AdaptiveLogSoftmaxWithLoss(
            in_features=0,  # Invalid: in_features must be > 0
            n_classes=10,
            cutoffs=[5],
            div_value=4.0,
            head_bias=False
        )
    assert "in_features" in str(exc_info.value).lower(), \
        f"Expected error about in_features, got: {exc_info.value}"
    
    # Test cutoffs value out of range (>= n_classes)
    with pytest.raises(ValueError) as exc_info:
        AdaptiveLogSoftmaxWithLoss(
            in_features=10,
            n_classes=20,
            cutoffs=[25],  # Invalid: 25 >= 20
            div_value=4.0,
            head_bias=False
        )
    assert "cutoff" in str(exc_info.value).lower() or "range" in str(exc_info.value).lower(), \
        f"Expected error about cutoff range, got: {exc_info.value}"
    
    # Test cutoffs value <= 0
    with pytest.raises(ValueError) as exc_info:
        AdaptiveLogSoftmaxWithLoss(
            in_features=10,
            n_classes=20,
            cutoffs=[0],  # Invalid: must be > 0
            div_value=4.0,
            head_bias=False
        )
    assert "cutoff" in str(exc_info.value).lower() or "positive" in str(exc_info.value).lower(), \
        f"Expected error about positive cutoff, got: {exc_info.value}"
    
    # Test div_value <= 0
    with pytest.raises(ValueError) as exc_info:
        AdaptiveLogSoftmaxWithLoss(
            in_features=10,
            n_classes=20,
            cutoffs=[5],
            div_value=0.0,  # Invalid: must be > 0
            head_bias=False
        )
    assert "div_value" in str(exc_info.value).lower() or "positive" in str(exc_info.value).lower(), \
        f"Expected error about positive div_value, got: {exc_info.value}"

def test_adaptive_softmax_log_prob_edge_cases():
    """Test edge cases for log_prob method."""
    # Test with single cutoff
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=5,
        n_classes=10,
        cutoffs=[5],
        div_value=2.0,
        head_bias=False
    )
    
    # Test non-batched input
    input_tensor = torch.randn(5)
    log_probs = model.log_prob(input_tensor)
    
    assert log_probs.shape == (10,), f"Expected shape (10,), got {log_probs.shape}"
    assert torch.all(torch.isfinite(log_probs)), "log_prob output contains non-finite values"
    
    # Test batched input
    input_tensor_batch = torch.randn(3, 5)
    log_probs_batch = model.log_prob(input_tensor_batch)
    
    assert log_probs_batch.shape == (3, 10), f"Expected shape (3, 10), got {log_probs_batch.shape}"
    assert torch.all(torch.isfinite(log_probs_batch)), "log_prob batch output contains non-finite values"
    
    # Test that probabilities sum to ~1
    probs_sum = torch.exp(log_probs_batch).sum(dim=1)
    for i in range(3):
        assert abs(probs_sum[i].item() - 1.0) < 1e-5, \
            f"Example {i}: sum of probabilities should be ~1, got {probs_sum[i].item()}"
# ==== BLOCK:FOOTER END ====