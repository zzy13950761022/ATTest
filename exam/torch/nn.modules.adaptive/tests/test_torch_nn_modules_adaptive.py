import math
import pytest
import torch
import torch.nn as nn
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions

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
# Basic smoke test for AdaptiveLogSoftmaxWithLoss

def test_adaptive_softmax_basic_functionality():
    """Basic smoke test to verify module can be imported and instantiated."""
    # Test basic instantiation
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=10,
        n_classes=100,
        cutoffs=[10, 50],
        div_value=4.0,
        head_bias=False
    )
    
    assert model is not None, "Module should be created successfully"
    assert model.in_features == 10, f"Expected in_features=10, got {model.in_features}"
    assert model.n_classes == 100, f"Expected n_classes=100, got {model.n_classes}"
    assert model.cutoffs == [10, 50], f"Expected cutoffs=[10, 50], got {model.cutoffs}"
    
    # Test forward pass
    input_tensor = torch.randn(2, 10)
    target_tensor = torch.randint(0, 100, (2,), dtype=torch.long)
    result = model(input_tensor, target_tensor)
    
    assert result.output.shape == (2,), f"Expected output shape (2,), got {result.output.shape}"
    assert result.loss.shape == (), f"Expected scalar loss, got {result.loss.shape}"
    assert torch.isfinite(result.loss), f"Loss should be finite, got {result.loss.item()}"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# Test parameter validation

def test_adaptive_softmax_parameter_validation():
    """Test parameter validation for AdaptiveLogSoftmaxWithLoss."""
    # Test valid parameters
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=5,
        n_classes=30,
        cutoffs=[5, 15],
        div_value=4.0,
        head_bias=False
    )
    assert model is not None, "Valid parameters should create module"
    
    # Test invalid n_classes (< 2)
    with pytest.raises(ValueError) as exc_info:
        AdaptiveLogSoftmaxWithLoss(
            in_features=10,
            n_classes=1,  # Invalid
            cutoffs=[5],
            div_value=4.0,
            head_bias=False
        )
    assert "n_classes" in str(exc_info.value).lower()
    
    # Test invalid in_features (<= 0)
    with pytest.raises(ValueError) as exc_info:
        AdaptiveLogSoftmaxWithLoss(
            in_features=0,  # Invalid
            n_classes=10,
            cutoffs=[5],
            div_value=4.0,
            head_bias=False
        )
    assert "in_features" in str(exc_info.value).lower()
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
# Test cutoffs validation

def test_adaptive_softmax_cutoffs_validation_basic():
    """Test basic cutoffs validation."""
    # Test valid cutoffs
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=5,
        n_classes=30,
        cutoffs=[5, 15],
        div_value=4.0,
        head_bias=False
    )
    assert model is not None, "Valid cutoffs should create module"
    
    # Test non-increasing cutoffs
    with pytest.raises(ValueError) as exc_info:
        AdaptiveLogSoftmaxWithLoss(
            in_features=5,
            n_classes=30,
            cutoffs=[30, 15],  # Invalid: non-increasing
            div_value=4.0,
            head_bias=False
        )
    assert "increasing" in str(exc_info.value).lower() or "sorted" in str(exc_info.value).lower()
    
    # Test duplicate cutoffs
    with pytest.raises(ValueError) as exc_info:
        AdaptiveLogSoftmaxWithLoss(
            in_features=5,
            n_classes=30,
            cutoffs=[5, 5, 15],  # Invalid: duplicate
            div_value=4.0,
            head_bias=False
        )
    assert "unique" in str(exc_info.value).lower() or "duplicate" in str(exc_info.value).lower()
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# Test helper methods

def test_adaptive_softmax_helper_methods():
    """Test helper methods like log_prob and predict."""
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=12,
        n_classes=80,
        cutoffs=[10, 30, 60],
        div_value=4.0,
        head_bias=False
    )
    
    # Test log_prob method exists
    assert hasattr(model, 'log_prob'), "Model should have log_prob method"
    assert callable(model.log_prob), "log_prob should be callable"
    
    # Test log_prob with batched input
    input_tensor = torch.randn(2, 12)
    log_probs = model.log_prob(input_tensor)
    
    assert isinstance(log_probs, torch.Tensor), "log_prob should return Tensor"
    assert log_probs.shape == (2, 80), f"Expected shape (2, 80), got {log_probs.shape}"
    assert torch.all(torch.isfinite(log_probs)), "log_prob output should be finite"
    
    # Test predict method if it exists
    if hasattr(model, 'predict') and callable(model.predict):
        predictions = model.predict(input_tensor)
        assert isinstance(predictions, torch.Tensor), "predict should return Tensor"
        assert predictions.shape == (2,), f"Expected shape (2,), got {predictions.shape}"
        assert predictions.dtype == torch.long, f"Expected long type, got {predictions.dtype}"
        assert torch.all(predictions >= 0), "Predictions should be >= 0"
        assert torch.all(predictions < 80), "Predictions should be < n_classes"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# Test device compatibility (basic)

def test_adaptive_softmax_device_compatibility():
    """Test basic device compatibility."""
    # Test CPU
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=6,
        n_classes=40,
        cutoffs=[5, 20],
        div_value=4.0,
        head_bias=False
    ).to("cpu")
    
    input_tensor = torch.randn(2, 6, device="cpu")
    target_tensor = torch.randint(0, 40, (2,), dtype=torch.long, device="cpu")
    result = model(input_tensor, target_tensor)
    
    assert result.output.device == torch.device("cpu"), "Output should be on CPU"
    assert result.loss.device == torch.device("cpu"), "Loss should be on CPU"
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# Additional basic tests

def test_adaptive_softmax_shape_handling():
    """Test shape handling for different input types."""
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=8,
        n_classes=50,
        cutoffs=[5, 20],
        div_value=2.0,
        head_bias=True
    )
    
    # Test batched input
    input_batch = torch.randn(3, 8)
    target_batch = torch.randint(0, 50, (3,), dtype=torch.long)
    result_batch = model(input_batch, target_batch)
    assert result_batch.output.shape == (3,), f"Expected shape (3,), got {result_batch.output.shape}"
    
    # Test non-batched input
    input_single = torch.randn(8)
    target_single = torch.randint(0, 50, (), dtype=torch.long)
    result_single = model(input_single, target_single)
    assert result_single.output.shape == (), f"Expected scalar output, got {result_single.output.shape}"
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# Test case: deferred placeholder
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# Test error handling

def test_adaptive_softmax_error_handling():
    """Test error handling for invalid inputs."""
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=10,
        n_classes=100,
        cutoffs=[10, 50],
        div_value=4.0,
        head_bias=False
    )
    
    # Test input shape mismatch
    input_tensor = torch.randn(3, 8)  # Wrong: 8 instead of 10
    target_tensor = torch.randint(0, 100, (3,), dtype=torch.long)
    
    with pytest.raises(RuntimeError):
        model(input_tensor, target_tensor)
    
    # Test target shape mismatch
    input_tensor2 = torch.randn(3, 10)  # Correct shape
    target_tensor2 = torch.randint(0, 100, (4,), dtype=torch.long)  # Wrong: 4 instead of 3
    
    with pytest.raises(RuntimeError):
        model(input_tensor2, target_tensor2)
    
    # Test target out of range
    input_tensor3 = torch.randn(2, 10)
    target_tensor3 = torch.tensor([-1, 100], dtype=torch.long)  # Invalid: -1 and 100 (n_classes=100)
    
    with pytest.raises(IndexError):
        model(input_tensor3, target_tensor3)
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Additional utilities and cleanup

def test_adaptive_softmax_repr():
    """Test string representation of AdaptiveLogSoftmaxWithLoss."""
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=10,
        n_classes=100,
        cutoffs=[10, 50],
        div_value=4.0,
        head_bias=False
    )
    
    repr_str = repr(model)
    assert "AdaptiveLogSoftmaxWithLoss" in repr_str
    assert "in_features=10" in repr_str
    assert "n_classes=100" in repr_str
    
    # Test that model can be printed without error
    print(f"Model representation: {repr_str}")

def test_adaptive_softmax_module_attributes():
    """Test that module has expected attributes."""
    model = AdaptiveLogSoftmaxWithLoss(
        in_features=10,
        n_classes=100,
        cutoffs=[10, 50],
        div_value=4.0,
        head_bias=False
    )
    
    # Check required attributes
    assert hasattr(model, 'head'), "Module should have head attribute"
    assert hasattr(model, 'tail'), "Module should have tail attribute"
    assert isinstance(model.tail, nn.ModuleList), "tail should be a ModuleList"
    
    # Check that head is a Linear layer
    assert isinstance(model.head, nn.Linear), "head should be a Linear layer"
    
    # Check parameter counts
    head_params = sum(p.numel() for p in model.head.parameters())
    assert head_params > 0, "Head should have parameters"
    
    # Check that model is a nn.Module
    assert isinstance(model, nn.Module), "AdaptiveLogSoftmaxWithLoss should be a nn.Module"

# Note: Comprehensive tests are now in the group-specific test files:
# - tests/test_torch_nn_modules_adaptive_g1.py (G1 group: core functionality)
# - tests/test_torch_nn_modules_adaptive_g2.py (G2 group: parameter validation and helper methods)
# ==== BLOCK:FOOTER END ====