"""
Test module for torch.nn.modules.activation - Group G3: Threshold and Piecewise Functions
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
Test module for torch.nn.modules.activation - Group G3: Threshold and Piecewise Functions
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

def assert_hardtanh_properties(output, input_tensor, min_val, max_val, test_name=""):
    """Assert Hardtanh-specific properties.
    
    Args:
        output: Hardtanh output tensor
        input_tensor: Input tensor
        min_val: Minimum value for clamping
        max_val: Maximum value for clamping
        test_name: Test name for error messages
    """
    # Check that output is clamped between min_val and max_val
    assert torch.all(output >= min_val), f"{test_name}: Output contains values below min_val={min_val}"
    assert torch.all(output <= max_val), f"{test_name}: Output contains values above max_val={max_val}"
    
    # Check that values within [min_val, max_val] are unchanged
    within_mask = (input_tensor >= min_val) & (input_tensor <= max_val)
    if torch.any(within_mask):
        assert torch.allclose(output[within_mask], input_tensor[within_mask], rtol=1e-5, atol=1e-8), \
            f"{test_name}: Values within bounds should not be modified"
    
    # Check that values below min_val are clamped to min_val
    below_mask = input_tensor < min_val
    if torch.any(below_mask):
        assert torch.all(output[below_mask] == min_val), \
            f"{test_name}: Values below min_val should be clamped to min_val"
    
    # Check that values above max_val are clamped to max_val
    above_mask = input_tensor > max_val
    if torch.any(above_mask):
        assert torch.all(output[above_mask] == max_val), \
            f"{test_name}: Values above max_val should be clamped to max_val"
    
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
# Test case: Softmax基础功能 (G2 group - placeholder)
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
@pytest.mark.parametrize("activation_params", [
    {
        "activation": "Hardtanh",
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False,
        "dtype": torch.float32,
        "device": "cpu",
        "shape": (2, 2),
        "input_range": "extended"
    }
])
def test_hardtanh_threshold(activation_params):
    """Test threshold function for Hardtanh activation.
    
    Test ID: TC-04
    Block ID: CASE_04
    Group: G3
    Assertion level: weak
    """
    # Extract parameters
    activation = activation_params["activation"]
    min_val = activation_params["min_val"]
    max_val = activation_params["max_val"]
    inplace = activation_params["inplace"]
    dtype = activation_params["dtype"]
    device = activation_params["device"]
    shape = activation_params["shape"]
    input_range = activation_params["input_range"]
    
    # Generate input tensor
    x = generate_input(shape, dtype, device, input_range)
    
    # Create activation module
    if activation == "Hardtanh":
        module = nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=inplace)
        # Oracle: torch.clamp
        oracle_func = lambda x: torch.clamp(x, min=min_val, max=max_val)
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
    
    # 4. Value clamping (Hardtanh specific)
    # Check that output is clamped between min_val and max_val
    assert torch.all(output >= min_val), f"Output contains values below min_val={min_val}"
    assert torch.all(output <= max_val), f"Output contains values above max_val={max_val}"
    
    # 5. Compare with oracle (torch.clamp)
    expected = oracle_func(x)
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-8), \
        f"Output doesn't match {activation} oracle (torch.clamp)"
    
    # 6. Test inplace behavior
    if inplace:
        # For inplace=True, output should be the same tensor as input (modified)
        # Note: nn.Hardtanh with inplace=True modifies the input
        # We need to test this separately
        x_copy = x.clone()
        module_inplace = nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=True)
        output_inplace = module_inplace(x_copy)
        # In inplace mode, the input is modified and returned
        assert x_copy.data_ptr() == output_inplace.data_ptr(), \
            "Inplace operation should return the same tensor"
        assert torch.allclose(output_inplace, expected, rtol=1e-5, atol=1e-8), \
            "Inplace output should match oracle"
    else:
        # For inplace=False, input should not be modified
        x_before = x.clone()
        _ = module(x)  # Forward pass
        assert torch.equal(x, x_before), "Non-inplace operation should not modify input"
    
    # 7. Test specific clamping behavior
    # Create test cases with known values
    test_cases = [
        # (input_value, expected_output)
        (-2.0, min_val),    # Below min_val
        (-1.0, -1.0),       # Equal to min_val
        (-0.5, -0.5),       # Within range
        (0.0, 0.0),         # Within range
        (0.5, 0.5),         # Within range
        (1.0, 1.0),         # Equal to max_val
        (2.0, max_val),     # Above max_val
    ]
    
    for input_val, expected_val in test_cases:
        test_tensor = torch.tensor([[input_val]], dtype=dtype, device=device)
        test_output = module(test_tensor)
        assert torch.allclose(test_output, torch.tensor([[expected_val]], dtype=dtype), 
                            rtol=1e-5, atol=1e-8), \
            f"Hardtanh({input_val}) should be {expected_val}, got {test_output.item()}"
    
    # 8. Test edge cases
    # Test with values exactly at boundaries
    boundary_tensor = torch.tensor([[min_val, max_val]], dtype=dtype, device=device)
    boundary_output = module(boundary_tensor)
    expected_boundary = torch.tensor([[min_val, max_val]], dtype=dtype, device=device)
    assert torch.allclose(boundary_output, expected_boundary, rtol=1e-5, atol=1e-8), \
        "Boundary values should remain unchanged"
    
    # 9. Test with all values within range
    within_tensor = torch.tensor([[-0.7, 0.3], [0.1, 0.9]], dtype=dtype, device=device)
    within_output = module(within_tensor)
    assert torch.allclose(within_output, within_tensor, rtol=1e-5, atol=1e-8), \
        "Values within range should not be modified"
    
    # 10. Test with all values outside range
    outside_tensor = torch.tensor([[-3.0, -2.5], [1.5, 2.0]], dtype=dtype, device=device)
    outside_output = module(outside_tensor)
    expected_outside = torch.tensor([[min_val, min_val], [max_val, max_val]], 
                                   dtype=dtype, device=device)
    assert torch.allclose(outside_output, expected_outside, rtol=1e-5, atol=1e-8), \
        "Values outside range should be clamped to boundaries"
    
    # 11. Test monotonicity
    # Hardtanh should be monotonic non-decreasing
    monotonic_test = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], 
                                 dtype=dtype, device=device)
    monotonic_output = module(monotonic_test)
    
    # Check that output is non-decreasing
    for i in range(len(monotonic_output) - 1):
        assert monotonic_output[i] <= monotonic_output[i + 1], \
            f"Hardtanh should be monotonic: output[{i}]={monotonic_output[i]} > output[{i+1}]={monotonic_output[i+1]}"
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
# Additional tests for G3 group activation functions (Threshold and Piecewise Functions)

def test_hardtanh_parameter_validation():
    """Test Hardtanh parameter validation."""
    # Test with valid parameters
    hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
    assert hardtanh.min_val == -1.0
    assert hardtanh.max_val == 1.0
    
    # Test with default parameters
    hardtanh_default = nn.Hardtanh()
    assert hardtanh_default.min_val == -1.0
    assert hardtanh_default.max_val == 1.0
    
    # Test with custom parameters
    hardtanh_custom = nn.Hardtanh(min_val=-2.5, max_val=3.0)
    assert hardtanh_custom.min_val == -2.5
    assert hardtanh_custom.max_val == 3.0
    
    # Test that max_val > min_val is required
    # Note: nn.Hardtanh uses assert, not ValueError, for parameter validation
    # So we need to catch AssertionError instead of ValueError
    with pytest.raises(AssertionError):
        nn.Hardtanh(min_val=1.0, max_val=-1.0)  # max_val < min_val
    
    with pytest.raises(AssertionError):
        nn.Hardtanh(min_val=1.0, max_val=1.0)  # max_val == min_val

def test_hardtanh_clamping_behavior():
    """Test Hardtanh clamping behavior."""
    # Create Hardtanh with specific bounds
    hardtanh = nn.Hardtanh(min_val=-2.0, max_val=3.0)
    
    # Test values within bounds
    x_within = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    output_within = hardtanh(x_within)
    expected_within = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    assert torch.allclose(output_within, expected_within, rtol=1e-5, atol=1e-8), \
        "Values within bounds should not be clamped"
    
    # Test values below min_val
    x_below = torch.tensor([-5.0, -3.0, -2.5])
    output_below = hardtanh(x_below)
    expected_below = torch.tensor([-2.0, -2.0, -2.0])  # All clamped to min_val
    assert torch.allclose(output_below, expected_below, rtol=1e-5, atol=1e-8), \
        "Values below min_val should be clamped to min_val"
    
    # Test values above max_val
    x_above = torch.tensor([3.5, 4.0, 5.0])
    output_above = hardtanh(x_above)
    expected_above = torch.tensor([3.0, 3.0, 3.0])  # All clamped to max_val
    assert torch.allclose(output_above, expected_above, rtol=1e-5, atol=1e-8), \
        "Values above max_val should be clamped to max_val"
    
    # Test mixed values
    x_mixed = torch.tensor([-3.0, -1.0, 0.0, 2.0, 4.0])
    output_mixed = hardtanh(x_mixed)
    expected_mixed = torch.tensor([-2.0, -1.0, 0.0, 2.0, 3.0])
    assert torch.allclose(output_mixed, expected_mixed, rtol=1e-5, atol=1e-8), \
        "Mixed values should be correctly clamped"

def test_relu6_basic():
    """Test ReLU6 basic functionality."""
    # ReLU6 is ReLU with max value of 6: min(max(0, x), 6)
    relu6 = nn.ReLU6()
    
    # Test negative values
    x_neg = torch.tensor([-2.0, -1.0, -0.5])
    output_neg = relu6(x_neg)
    expected_neg = torch.tensor([0.0, 0.0, 0.0])
    assert torch.allclose(output_neg, expected_neg, rtol=1e-5, atol=1e-8), \
        "Negative values should become 0"
    
    # Test values between 0 and 6
    x_mid = torch.tensor([0.0, 1.0, 3.0, 5.0])
    output_mid = relu6(x_mid)
    expected_mid = torch.tensor([0.0, 1.0, 3.0, 5.0])
    assert torch.allclose(output_mid, expected_mid, rtol=1e-5, atol=1e-8), \
        "Values between 0 and 6 should remain unchanged"
    
    # Test values above 6
    x_above = torch.tensor([6.0, 7.0, 10.0])
    output_above = relu6(x_above)
    expected_above = torch.tensor([6.0, 6.0, 6.0])
    assert torch.allclose(output_above, expected_above, rtol=1e-5, atol=1e-8), \
        "Values above 6 should be clamped to 6"
    
    # Test boundary values
    x_boundary = torch.tensor([-0.0, 0.0, 6.0, 6.0001])
    output_boundary = relu6(x_boundary)
    expected_boundary = torch.tensor([0.0, 0.0, 6.0, 6.0])
    assert torch.allclose(output_boundary, expected_boundary, rtol=1e-5, atol=1e-8), \
        "Boundary values should be handled correctly"

def test_celu_basic():
    """Test CELU (Continuously Differentiable Exponential Linear Unit) basic functionality."""
    # CELU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    # Default alpha = 1.0
    celu = nn.CELU()
    
    # Test positive values
    x_pos = torch.tensor([0.0, 1.0, 2.0])
    output_pos = celu(x_pos)
    expected_pos = torch.tensor([0.0, 1.0, 2.0])  # max(0, x) part
    assert torch.allclose(output_pos, expected_pos, rtol=1e-5, atol=1e-8), \
        "Positive values should remain unchanged"
    
    # Test negative values
    x_neg = torch.tensor([-1.0, -2.0])
    output_neg = celu(x_neg)
    # alpha * (exp(x/alpha) - 1) = 1 * (exp(x) - 1)
    expected_neg = torch.tensor([math.exp(-1.0) - 1.0, math.exp(-2.0) - 1.0])
    assert torch.allclose(output_neg, expected_neg, rtol=1e-5, atol=1e-8), \
        "Negative values should follow CELU formula"
    
    # Test with custom alpha
    alpha = 2.0
    celu_custom = nn.CELU(alpha=alpha)
    x_custom = torch.tensor([-1.0, -2.0])
    output_custom = celu_custom(x_custom)
    # alpha * (exp(x/alpha) - 1) = 2 * (exp(x/2) - 1)
    expected_custom = torch.tensor([
        2.0 * (math.exp(-0.5) - 1.0),
        2.0 * (math.exp(-1.0) - 1.0)
    ])
    assert torch.allclose(output_custom, expected_custom, rtol=1e-5, atol=1e-8), \
        "CELU with custom alpha should follow formula"

def test_selu_basic():
    """Test SELU (Scaled Exponential Linear Unit) basic functionality."""
    # SELU has fixed parameters: alpha ≈ 1.67326, scale ≈ 1.0507
    selu = nn.SELU()
    
    # Test positive values
    x_pos = torch.tensor([0.0, 1.0, 2.0])
    output_pos = selu(x_pos)
    scale = 1.0507009873554804934193349852946
    expected_pos = scale * x_pos
    assert torch.allclose(output_pos, expected_pos, rtol=1e-5, atol=1e-8), \
        "Positive values should be scaled by 1.0507"
    
    # Test negative values
    x_neg = torch.tensor([-1.0, -2.0])
    output_neg = selu(x_neg)
    alpha = 1.6732632423543772848170429916717
    expected_neg = scale * alpha * (torch.exp(x_neg) - 1.0)
    assert torch.allclose(output_neg, expected_neg, rtol=1e-5, atol=1e-8), \
        "Negative values should follow SELU formula"
    
    # Test that SELU is self-normalizing (output mean ~0, variance ~1 for normalized inputs)
    # This is more of a property than a strict assertion
    x_normalized = torch.randn(1000)  # Standard normal
    output_normalized = selu(x_normalized)
    output_mean = torch.mean(output_normalized)
    output_std = torch.std(output_normalized)
    
    # SELU should produce outputs with mean close to 0 and std close to 1
    # (for normalized inputs)
    assert abs(output_mean.item()) < 0.1, f"SELU output mean should be close to 0, got {output_mean.item()}"
    assert 0.9 < output_std.item() < 1.1, f"SELU output std should be close to 1, got {output_std.item()}"

def test_gelu_basic():
    """Test GELU (Gaussian Error Linear Unit) basic functionality."""
    # GELU(x) = x * Φ(x) where Φ is the CDF of standard Gaussian
    # PyTorch uses a more accurate approximation than the tanh formula
    gelu = nn.GELU()
    
    # Test at x = 0
    x_zero = torch.tensor([0.0])
    output_zero = gelu(x_zero)
    assert torch.allclose(output_zero, torch.tensor([0.0]), rtol=1e-5, atol=1e-8), \
        "GELU(0) should be 0"
    
    # Test positive values
    x_pos = torch.tensor([1.0, 2.0])
    output_pos = gelu(x_pos)
    # GELU should be monotonic increasing
    assert output_pos[0] < output_pos[1], "GELU should be monotonic increasing"
    
    # Test negative values
    x_neg = torch.tensor([-1.0, -2.0])
    output_neg = gelu(x_neg)
    # GELU should output negative values for negative inputs
    assert torch.all(output_neg < 0), "GELU should output negative values for negative inputs"
    
    # Test symmetry property: GELU is not symmetric, but check specific values
    # GELU(-x) ≠ -GELU(x) in general
    
    # Compare with PyTorch's functional GELU implementation as reference
    # This is more reliable than the approximate formula
    x_test = torch.tensor([0.5, -0.5, 1.5, -1.5])
    output_test = gelu(x_test)
    
    # Use torch.nn.functional.gelu as reference
    expected_test = F.gelu(x_test)
    
    # Use more relaxed tolerance since different implementations may have small differences
    assert torch.allclose(output_test, expected_test, rtol=1e-5, atol=1e-6), \
        "nn.GELU should match F.gelu"
    
    # Test with approximate formula for educational purposes (with relaxed tolerance)
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    approx_gelu = lambda x: 0.5 * x * (1.0 + torch.tanh(sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))))
    approx_test = approx_gelu(x_test)
    
    # The approximate formula may differ from PyTorch's implementation
    # Use very relaxed tolerance for educational comparison
    assert torch.allclose(output_test, approx_test, rtol=1e-2, atol=1e-3), \
        "GELU should be roughly similar to approximate formula"

def test_module_attributes_g3():
    """Test that G3 activation modules have correct attributes."""
    # Test Hardtanh
    hardtanh = nn.Hardtanh(min_val=-2.0, max_val=3.0)
    assert hasattr(hardtanh, 'min_val'), "Hardtanh should have 'min_val' attribute"
    assert hasattr(hardtanh, 'max_val'), "Hardtanh should have 'max_val' attribute"
    assert hardtanh.min_val == -2.0, f"Hardtanh min_val should be -2.0, got {hardtanh.min_val}"
    assert hardtanh.max_val == 3.0, f"Hardtanh max_val should be 3.0, got {hardtanh.max_val}"
    
    # Test ReLU6 (no special parameters beyond ReLU)
    relu6 = nn.ReLU6()
    # ReLU6 inherits from ReLU
    
    # Test CELU
    celu = nn.CELU(alpha=2.0)
    assert hasattr(celu, 'alpha'), "CELU should have 'alpha' attribute"
    assert celu.alpha == 2.0, f"CELU alpha should be 2.0, got {celu.alpha}"
    
    # Test SELU (fixed parameters)
    selu = nn.SELU()
    # SELU has fixed parameters
    
    # Test GELU (no parameters in basic version)
    gelu = nn.GELU()
    # GELU has no parameters in basic version

def test_module_str_representation_g3():
    """Test string representation of G3 activation modules."""
    hardtanh = nn.Hardtanh(min_val=-2.0, max_val=3.0)
    assert "Hardtanh" in str(hardtanh), "Hardtanh string representation should contain 'Hardtanh'"
    assert "min_val=-2.0" in str(hardtanh).replace(" ", ""), "Hardtanh should show min_val parameter"
    assert "max_val=3.0" in str(hardtanh).replace(" ", ""), "Hardtanh should show max_val parameter"
    
    relu6 = nn.ReLU6()
    assert "ReLU6" in str(relu6), "ReLU6 string representation should contain 'ReLU6'"
    
    celu = nn.CELU(alpha=2.0)
    assert "CELU" in str(celu), "CELU string representation should contain 'CELU'"
    assert "alpha=2.0" in str(celu).replace(" ", ""), "CELU should show alpha parameter"
    
    selu = nn.SELU()
    assert "SELU" in str(selu), "SELU string representation should contain 'SELU'"
    
    gelu = nn.GELU()
    assert "GELU" in str(gelu), "GELU string representation should contain 'GELU'"

def test_inplace_behavior_g3():
    """Test inplace behavior for G3 activation functions that support it."""
    # Hardtanh supports inplace
    x_hardtanh = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0])
    x_hardtanh_copy = x_hardtanh.clone()
    
    hardtanh_inplace = nn.Hardtanh(min_val=-2.0, max_val=3.0, inplace=True)
    hardtanh_regular = nn.Hardtanh(min_val=-2.0, max_val=3.0, inplace=False)
    
    # Test regular (non-inplace)
    output_regular = hardtanh_regular(x_hardtanh)
    assert not torch.equal(x_hardtanh, output_regular), "Non-inplace Hardtanh should not modify input"
    
    # Test inplace
    output_inplace = hardtanh_inplace(x_hardtanh_copy)
    assert torch.equal(x_hardtanh_copy, output_inplace), "Inplace Hardtanh should modify input"
    assert torch.equal(x_hardtanh_copy, output_regular), "Inplace and non-inplace should produce same output"
    
    # ReLU6 does not have inplace parameter in nn.ReLU6 (but nn.ReLU does)
    # CELU, SELU, GELU don't have inplace parameter

# Cleanup and teardown
def teardown_module():
    """Cleanup after all tests."""
    # Clear any cached tensors
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
# ==== BLOCK:FOOTER END ====