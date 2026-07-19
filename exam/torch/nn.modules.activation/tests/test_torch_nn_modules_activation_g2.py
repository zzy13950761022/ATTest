"""
Test module for torch.nn.modules.activation - Group G2: Softmax and Normalization Functions
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)

# ==== BLOCK:HEADER START ====
"""
Test module for torch.nn.modules.activation - Group G2: Softmax and Normalization Functions
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)

# Common test utilities
def generate_input(shape, dtype, device, input_range="normal"):
    """Generate test input tensor based on range specification.
    
    Args:
        shape: Tensor shape
        dtype: Tensor data type
        device: Tensor device
        input_range: One of "normal", "extended", "mixed", "extreme"
    
    Returns:
        torch.Tensor: Generated input tensor
    """
    if input_range == "normal":
        # Normal distribution with mean 0, std 1
        tensor = torch.randn(shape, dtype=dtype, device=device)
    elif input_range == "extended":
        # Extended range including values outside typical bounds
        tensor = torch.randn(shape, dtype=dtype, device=device) * 3.0
    elif input_range == "mixed":
        # Mix of positive and negative values with explicit negatives
        tensor = torch.randn(shape, dtype=dtype, device=device)
        # Make some values explicitly negative
        if tensor.numel() > 0:
            mask = torch.rand(tensor.shape) < 0.3
            tensor[mask] = -torch.abs(tensor[mask]) - 0.5
    elif input_range == "extreme":
        # Extreme values including very large/small numbers
        tensor = torch.randn(shape, dtype=dtype, device=device) * 10.0
        # Add some extreme values
        if tensor.numel() > 0:
            flat_tensor = tensor.view(-1)
            # Set first few elements to extreme values
            if len(flat_tensor) >= 1:
                flat_tensor[0] = 100.0  # Very large positive
            if len(flat_tensor) >= 2:
                flat_tensor[1] = -100.0  # Very large negative
            if len(flat_tensor) >= 3:
                flat_tensor[2] = 1e6  # Extremely large positive
            if len(flat_tensor) >= 4:
                flat_tensor[3] = -1e6  # Extremely large negative
    else:
        raise ValueError(f"Unknown input_range: {input_range}")
    return tensor

def assert_tensor_properties(output, expected_shape, expected_dtype, test_name=""):
    """Assert basic tensor properties.
    
    Args:
        output: Output tensor to check
        expected_shape: Expected shape
        expected_dtype: Expected data type
        test_name: Test name for error messages
    
    Returns:
        bool: True if all assertions pass
    """
    assert output.shape == expected_shape, f"{test_name}: Shape mismatch: {output.shape} != {expected_shape}"
    assert output.dtype == expected_dtype, f"{test_name}: Dtype mismatch: {output.dtype} != {expected_dtype}"
    assert torch.all(torch.isfinite(output)), f"{test_name}: Output contains non-finite values"
    return True

def assert_softmax_properties(output, dim, test_name=""):
    """Assert Softmax-specific properties.
    
    Args:
        output: Softmax output tensor
        dim: Dimension along which Softmax was applied
        test_name: Test name for error messages
    """
    # Check non-negative values
    assert torch.all(output >= 0), f"{test_name}: Softmax output contains negative values"
    
    # Check sum to 1 along specified dimension
    sums = torch.sum(output, dim=dim)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-8), \
        f"{test_name}: Softmax outputs do not sum to 1 along dim={dim}"
    
    return True

# Common fixtures
@pytest.fixture
def cpu_device():
    """Fixture for CPU device."""
    return torch.device("cpu")

@pytest.fixture
def float32_dtype():
    """Fixture for float32 dtype."""
    return torch.float32

@pytest.fixture
def float64_dtype():
    """Fixture for float64 dtype."""
    return torch.float64

@pytest.fixture
def normal_input_data():
    """Fixture for normal input data."""
    def _generate(shape, dtype=torch.float32, device="cpu"):
        return generate_input(shape, dtype, device, "normal")
    return _generate

@pytest.fixture
def extended_input_data():
    """Fixture for extended input data."""
    def _generate(shape, dtype=torch.float32, device="cpu"):
        return generate_input(shape, dtype, device, "extended")
    return _generate

@pytest.fixture
def mixed_input_data():
    """Fixture for mixed input data."""
    def _generate(shape, dtype=torch.float32, device="cpu"):
        return generate_input(shape, dtype, device, "mixed")
    return _generate

@pytest.fixture
def extreme_input_data():
    """Fixture for extreme input data."""
    def _generate(shape, dtype=torch.float32, device="cpu"):
        return generate_input(shape, dtype, device, "extreme")
    return _generate
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# Test case: ReLU基础正向传播 (G1 group - placeholder)
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# Test case: Sigmoid与Tanh基础测试 (G1 group - placeholder)
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize("activation_params", [
    {
        "activation": "Softmax",
        "dim": 1,
        "dtype": torch.float32,
        "device": "cpu",
        "shape": (2, 3, 4),
        "input_range": "normal"
    },
    # Parameter extension from test_plan.json
    {
        "activation": "Softmax",
        "dim": -1,
        "dtype": torch.float32,
        "device": "cpu",
        "shape": (5,),
        "input_range": "normal"
    },
    # Add extreme input test case
    {
        "activation": "Softmax",
        "dim": 1,
        "dtype": torch.float32,
        "device": "cpu",
        "shape": (2, 3),
        "input_range": "extreme"
    },
    # Add monotonicity test case with specific shape
    {
        "activation": "Softmax",
        "dim": 0,
        "dtype": torch.float32,
        "device": "cpu",
        "shape": (4,),
        "input_range": "mixed"
    }
])
def test_softmax_basic(activation_params):
    """Test basic functionality for Softmax activation function.
    
    Test ID: TC-03
    Block ID: CASE_03
    Group: G2
    Assertion level: weak
    """
    # Extract parameters
    activation = activation_params["activation"]
    dim = activation_params["dim"]
    dtype = activation_params["dtype"]
    device = activation_params["device"]
    shape = activation_params["shape"]
    input_range = activation_params["input_range"]
    
    # Generate input tensor
    x = generate_input(shape, dtype, device, input_range)
    
    # Create activation module
    if activation == "Softmax":
        module = nn.Softmax(dim=dim)
        oracle_func = lambda x: F.softmax(x, dim=dim)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    # Forward pass
    output = module(x)
    
    # Weak assertions
    # 1. Shape match
    assert output.shape == shape, f"Output shape mismatch: {output.shape} != {shape}"
    
    # 2. Dtype match
    assert output.dtype == dtype, f"Output dtype mismatch: {output.dtype} != {dtype}"
    
    # 3. Finite values
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
    
    # 4. Non-negative values (Softmax specific)
    assert torch.all(output >= 0), "Softmax output contains negative values"
    
    # 5. Sum to 1 along specified dimension
    sums = torch.sum(output, dim=dim)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-8), \
        f"Softmax outputs do not sum to 1 along dim={dim}"
    
    # 6. Compare with oracle (torch.nn.functional.softmax)
    expected = oracle_func(x)
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-8), \
        f"Output doesn't match {activation} oracle"
    
    # 7. Enhanced monotonicity property test
    # Softmax preserves ordering: if x_i > x_j, then softmax(x)_i > softmax(x)_j
    # Test for 1D tensors
    if len(shape) == 1:
        # Create indices for comparison
        for i in range(shape[0]):
            for j in range(i + 1, shape[0]):
                if x[i] > x[j]:
                    assert output[i] > output[j], \
                        f"Softmax should preserve ordering: x[{i}]={x[i]} > x[{j}]={x[j]}, but output[{i}]={output[i]} <= output[{j}]={output[j]}"
                elif x[i] < x[j]:
                    assert output[i] < output[j], \
                        f"Softmax should preserve ordering: x[{i}]={x[i]} < x[{j}]={x[j]}, but output[{i}]={output[i]} >= output[{j}]={output[j]}"
                # If equal, outputs should be equal (within tolerance)
                else:
                    assert torch.allclose(output[i], output[j], rtol=1e-5, atol=1e-8), \
                        f"Softmax should give equal outputs for equal inputs: x[{i}]={x[i]} == x[{j}]={x[j]}, but output[{i}]={output[i]} != output[{j}]={output[j]}"
    
    # 8. Enhanced numerical stability for extreme inputs
    if input_range == "extreme":
        # Even with extreme inputs, output should be finite and valid
        assert torch.all(torch.isfinite(output)), "Softmax should handle extreme inputs gracefully"
        
        # Sum should still be 1 (within numerical tolerance)
        sums_extreme = torch.sum(output, dim=dim)
        assert torch.allclose(sums_extreme, torch.ones_like(sums_extreme), rtol=1e-4, atol=1e-6), \
            f"Softmax with extreme inputs should sum to 1 along dim={dim}"
        
        # Check for NaN or inf in output
        assert not torch.any(torch.isnan(output)), "Softmax output should not contain NaN for extreme inputs"
        assert not torch.any(torch.isinf(output)), "Softmax output should not contain inf for extreme inputs"
        
        # Test with specific extreme values
        if shape == (2, 3):  # Our extreme test case shape
            # Verify that very large values don't cause overflow
            assert torch.all(output > 0), "Softmax output should be positive even for extreme inputs"
            assert torch.all(output < 1.0), "Softmax output should be less than 1 even for extreme inputs"
    
    # 9. Test that output values are in [0, 1] range
    assert torch.all(output >= 0) and torch.all(output <= 1.0), \
        "Softmax output values should be in [0, 1] range"
    
    # 10. Test invariance to constant addition
    # Softmax(x + c) = Softmax(x) for any constant c
    if len(shape) > 0:
        c = 10.0  # Arbitrary constant
        x_plus_c = x + c
        output_plus_c = module(x_plus_c)
        assert torch.allclose(output, output_plus_c, rtol=1e-5, atol=1e-8), \
            "Softmax should be invariant to constant addition"
        
        # Test with negative constant as well
        c_neg = -5.0
        x_plus_c_neg = x + c_neg
        output_plus_c_neg = module(x_plus_c_neg)
        assert torch.allclose(output, output_plus_c_neg, rtol=1e-5, atol=1e-8), \
            "Softmax should be invariant to negative constant addition"
    
    # 11. Test with all equal inputs (should give uniform distribution)
    if len(shape) > 0:
        # Create a tensor with all equal values
        equal_value = 2.5  # Arbitrary value
        x_equal = torch.full(shape, equal_value, dtype=dtype, device=device)
        output_equal = module(x_equal)
        
        # All outputs along the softmax dimension should be equal
        # For 1D tensors, all outputs should be equal
        if len(shape) == 1:
            expected_value = 1.0 / shape[0]
            assert torch.allclose(output_equal, torch.full(shape, expected_value, dtype=dtype), 
                                rtol=1e-5, atol=1e-8), \
                f"Softmax of equal values should give uniform distribution: 1/{shape[0]}"
        
        # For multi-dimensional tensors, check along the softmax dimension
        else:
            # Take a slice to check
            if dim == 0 and shape[0] > 1:
                # Compare first two elements along dimension 0
                assert torch.allclose(output_equal[0], output_equal[1], rtol=1e-5, atol=1e-8), \
                    "Softmax of equal values should give equal outputs along softmax dimension"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# Test case: Hardtanh阈值函数 (G3 group - placeholder)
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# Test case: LeakyReLU参数化测试 (G1 deferred - placeholder)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# Test case: Deferred test case (placeholder)
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# Test case: Deferred test case (placeholder)
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# Test case: Deferred test case (placeholder)
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:CASE_09 START ====
# Test case: Deferred test case (placeholder)
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:CASE_10 START ====
# Test case: Deferred test case (placeholder)
# ==== BLOCK:CASE_10 END ====

# ==== BLOCK:CASE_11 START ====
# Test case: Deferred test case (placeholder)
# ==== BLOCK:CASE_11 END ====

# ==== BLOCK:CASE_12 START ====
# Test case: Deferred test case (placeholder)
# ==== BLOCK:CASE_12 END ====

# ==== BLOCK:CASE_13 START ====
# Test case: Deferred test case (placeholder)
# ==== BLOCK:CASE_13 END ====

# ==== BLOCK:FOOTER START ====
# Additional tests for G2 group activation functions

def test_softmax_dim_validation():
    """Test Softmax dimension validation."""
    # Test with valid dimension
    softmax = nn.Softmax(dim=0)
    x = torch.randn(3, 4)
    output = softmax(x)
    assert output.shape == (3, 4)
    
    # Test with negative dimension
    softmax_neg = nn.Softmax(dim=-1)
    output_neg = softmax_neg(x)
    assert output_neg.shape == (3, 4)
    
    # Test that sum along dim=-1 (last dimension) equals 1
    sums = torch.sum(output_neg, dim=-1)
    assert torch.allclose(sums, torch.ones(3), rtol=1e-5, atol=1e-8)
    
    # Test with invalid dimension should raise error during forward pass
    softmax_invalid = nn.Softmax(dim=5)  # dim > input.ndim
    with pytest.raises(IndexError):
        softmax_invalid(x)

def test_softmax_edge_cases():
    """Test Softmax with edge case inputs."""
    # Test with all zeros
    x_zeros = torch.zeros(2, 3)
    softmax = nn.Softmax(dim=1)
    output = softmax(x_zeros)
    
    # When all inputs are equal, softmax should give uniform distribution
    expected_uniform = torch.full((2, 3), 1.0/3.0)
    assert torch.allclose(output, expected_uniform, rtol=1e-5, atol=1e-8), \
        "Softmax of zeros should give uniform distribution"
    
    # Test with all equal non-zero values
    x_constant = torch.full((2, 3), 5.0)
    output_constant = softmax(x_constant)
    assert torch.allclose(output_constant, expected_uniform, rtol=1e-5, atol=1e-8), \
        "Softmax of constant values should give uniform distribution"
    
    # Test with very large negative values (numerical stability)
    x_large_neg = torch.tensor([[-1000.0, -1001.0, -1002.0]])
    output_large_neg = softmax(x_large_neg)
    # Should still sum to 1
    sums = torch.sum(output_large_neg, dim=1)
    assert torch.allclose(sums, torch.ones(1), rtol=1e-5, atol=1e-8), \
        "Softmax with large negative values should still sum to 1"

def test_logsoftmax_basic():
    """Test LogSoftmax basic functionality."""
    # LogSoftmax is log(Softmax(x))
    x = torch.randn(2, 3)
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    
    softmax_output = softmax(x)
    logsoftmax_output = logsoftmax(x)
    
    # Verify relationship: log(softmax(x)) = logsoftmax(x)
    expected_logsoftmax = torch.log(softmax_output)
    assert torch.allclose(logsoftmax_output, expected_logsoftmax, rtol=1e-5, atol=1e-8), \
        "LogSoftmax should equal log(Softmax)"
    
    # Verify numerical properties
    assert torch.all(torch.isfinite(logsoftmax_output)), "LogSoftmax output should be finite"
    
    # For valid probabilities, logsoftmax values should be <= 0
    # (since probabilities are <= 1, their log is <= 0)
    assert torch.all(logsoftmax_output <= 0), "LogSoftmax values should be <= 0"

def test_softmin_basic():
    """Test Softmin basic functionality."""
    # Softmin(x) = Softmax(-x)
    x = torch.randn(2, 3)
    softmin = nn.Softmin(dim=1)
    softmax = nn.Softmax(dim=1)
    
    softmin_output = softmin(x)
    softmax_neg_output = softmax(-x)
    
    # Verify relationship
    assert torch.allclose(softmin_output, softmax_neg_output, rtol=1e-5, atol=1e-8), \
        "Softmin(x) should equal Softmax(-x)"
    
    # Verify basic properties
    assert torch.all(softmin_output >= 0), "Softmin output should be non-negative"
    sums = torch.sum(softmin_output, dim=1)
    assert torch.allclose(sums, torch.ones(2), rtol=1e-5, atol=1e-8), \
        "Softmin outputs should sum to 1"

def test_softsign_basic():
    """Test Softsign basic functionality."""
    # Softsign(x) = x / (1 + |x|)
    x = torch.randn(2, 3)
    softsign = nn.Softsign()
    output = softsign(x)
    
    # Verify formula
    expected = x / (1 + torch.abs(x))
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-8), \
        "Softsign doesn't match mathematical formula"
    
    # Verify value range: Softsign(x) ∈ (-1, 1)
    assert torch.all(output > -1.0) and torch.all(output < 1.0), \
        "Softsign output should be in (-1, 1)"
    
    # Verify special values
    x_zero = torch.zeros(1)
    output_zero = softsign(x_zero)
    assert torch.allclose(output_zero, torch.zeros(1), rtol=1e-7), \
        "Softsign(0) should be 0"
    
    # Test with large values (approaches ±1)
    x_large = torch.tensor([100.0, -100.0])
    output_large = softsign(x_large)
    assert torch.allclose(output_large, torch.tensor([0.990099, -0.990099]), rtol=1e-4), \
        "Softsign of large values should approach ±1"

def test_module_attributes_g2():
    """Test that G2 activation modules have correct attributes."""
    # Test Softmax
    softmax = nn.Softmax(dim=1)
    assert hasattr(softmax, 'dim'), "Softmax should have 'dim' attribute"
    assert softmax.dim == 1, f"Softmax dim should be 1, got {softmax.dim}"
    
    # Test LogSoftmax
    logsoftmax = nn.LogSoftmax(dim=-1)
    assert hasattr(logsoftmax, 'dim'), "LogSoftmax should have 'dim' attribute"
    assert logsoftmax.dim == -1, f"LogSoftmax dim should be -1, got {logsoftmax.dim}"
    
    # Test Softmin
    softmin = nn.Softmin(dim=0)
    assert hasattr(softmin, 'dim'), "Softmin should have 'dim' attribute"
    assert softmin.dim == 0, f"Softmin dim should be 0, got {softmin.dim}"
    
    # Test Softsign (no special parameters)
    softsign = nn.Softsign()
    # Softsign has no special constructor parameters

def test_module_str_representation_g2():
    """Test string representation of G2 activation modules."""
    softmax = nn.Softmax(dim=1)
    assert "Softmax" in str(softmax), "Softmax string representation should contain 'Softmax'"
    assert "dim=1" in str(softmax), "Softmax string representation should show dim parameter"
    
    logsoftmax = nn.LogSoftmax(dim=-1)
    assert "LogSoftmax" in str(logsoftmax), "LogSoftmax string representation should contain 'LogSoftmax'"
    
    softmin = nn.Softmin(dim=0)
    assert "Softmin" in str(softmin), "Softmin string representation should contain 'Softmin'"
    
    softsign = nn.Softsign()
    assert "Softsign" in str(softsign), "Softsign string representation should contain 'Softsign'"

# Cleanup and teardown
def teardown_module():
    """Cleanup after all tests."""
    # Clear any cached tensors
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
# ==== BLOCK:FOOTER END ====