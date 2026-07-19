"""
Test module for torch.nn.modules.activation - Group G1: Basic Activation Functions
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
Test module for torch.nn.modules.activation - Group G1: Basic Activation Functions
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
    """Generate test input tensor based on range specification."""
    if input_range == "normal":
        # Normal distribution with mean 0, std 1
        tensor = torch.randn(shape, dtype=dtype, device=device)
    elif input_range == "extended":
        # Extended range including values outside typical bounds
        tensor = torch.randn(shape, dtype=dtype, device=device) * 3.0
    elif input_range == "mixed":
        # Mix of positive and negative values
        tensor = torch.randn(shape, dtype=dtype, device=device)
        # Make some values explicitly negative
        mask = torch.rand(shape) < 0.3
        tensor[mask] = -torch.abs(tensor[mask]) - 0.5
    elif input_range == "extreme":
        # Extreme values including very large/small numbers
        tensor = torch.randn(shape, dtype=dtype, device=device) * 10.0
        # Add some extreme values
        if tensor.numel() > 0:
            tensor.view(-1)[0] = 100.0
            tensor.view(-1)[1] = -100.0
    else:
        raise ValueError(f"Unknown input_range: {input_range}")
    return tensor

def assert_tensor_properties(output, expected_shape, expected_dtype, test_name=""):
    """Assert basic tensor properties."""
    assert output.shape == expected_shape, f"{test_name}: Shape mismatch: {output.shape} != {expected_shape}"
    assert output.dtype == expected_dtype, f"{test_name}: Dtype mismatch: {output.dtype} != {expected_dtype}"
    assert torch.all(torch.isfinite(output)), f"{test_name}: Output contains non-finite values"
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
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize("activation_params", [
    {
        "activation": "ReLU",
        "inplace": False,
        "dtype": torch.float32,
        "device": "cpu",
        "shape": (2, 3, 4),
        "input_range": "normal"
    },
    # Parameter extensions from test_plan.json
    {
        "activation": "ReLU",
        "inplace": True,
        "dtype": torch.float32,
        "device": "cpu",
        "shape": (2, 3, 4),
        "input_range": "normal"
    },
    {
        "activation": "ReLU",
        "inplace": False,
        "dtype": torch.float64,
        "device": "cpu",
        "shape": (4, 5),
        "input_range": "normal"
    }
])
def test_relu_basic_forward(activation_params):
    """Test basic forward propagation for ReLU activation function.
    
    Test ID: TC-01
    Block ID: CASE_01
    Group: G1
    Assertion level: weak
    """
    # Extract parameters
    activation = activation_params["activation"]
    inplace = activation_params["inplace"]
    dtype = activation_params["dtype"]
    device = activation_params["device"]
    shape = activation_params["shape"]
    input_range = activation_params["input_range"]
    
    # Generate input tensor
    x = generate_input(shape, dtype, device, input_range)
    
    # Create activation module
    if activation == "ReLU":
        module = nn.ReLU(inplace=inplace)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    # Store original input for comparison if inplace=False
    if not inplace:
        x_original = x.clone()
    
    # Forward pass
    output = module(x)
    
    # Weak assertions
    # 1. Shape match
    assert output.shape == shape, f"Output shape mismatch: {output.shape} != {shape}"
    
    # 2. Dtype match
    assert output.dtype == dtype, f"Output dtype mismatch: {output.dtype} != {dtype}"
    
    # 3. Finite values
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
    
    # 4. Non-negative values (ReLU specific)
    assert torch.all(output >= 0), "ReLU output contains negative values"
    
    # 5. Inplace behavior check
    if inplace:
        # When inplace=True, input should be modified
        assert torch.allclose(x, output, rtol=1e-5, atol=1e-8), \
            "Inplace operation failed: input not equal to output"
    else:
        # When inplace=False, input should remain unchanged
        assert torch.allclose(x, x_original, rtol=1e-5, atol=1e-8), \
            "Non-inplace operation modified input"
    
    # 6. Compare with oracle (torch.nn.functional.relu)
    expected = F.relu(x_original if not inplace else x, inplace=False)
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-8), \
        "Output doesn't match functional.relu"
    
    # 7. Verify ReLU properties: max(0, x)
    # For each element, output should be max(0, input)
    if not inplace:
        manual_relu = torch.where(x_original > 0, x_original, torch.zeros_like(x_original))
        assert torch.allclose(output, manual_relu, rtol=1e-5, atol=1e-8), \
            "Output doesn't match manual ReLU calculation"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("activation_params", [
    {
        "activation": "Sigmoid",
        "dtype": torch.float32,
        "device": "cpu",
        "shape": (3, 2),
        "input_range": "normal"
    },
    {
        "activation": "Tanh",
        "dtype": torch.float32,
        "device": "cpu",
        "shape": (3, 2),
        "input_range": "normal"
    },
    # Parameter extension from test_plan.json
    {
        "activation": "Sigmoid",
        "dtype": torch.float64,
        "device": "cpu",
        "shape": (1, 10),
        "input_range": "extreme"
    }
])
def test_sigmoid_tanh_basic(activation_params):
    """Test basic functionality for Sigmoid and Tanh activation functions.
    
    Test ID: TC-02
    Block ID: CASE_02
    Group: G1
    Assertion level: weak
    """
    # Extract parameters
    activation = activation_params["activation"]
    dtype = activation_params["dtype"]
    device = activation_params["device"]
    shape = activation_params["shape"]
    input_range = activation_params["input_range"]
    
    # Generate input tensor
    x = generate_input(shape, dtype, device, input_range)
    
    # Create activation module
    if activation == "Sigmoid":
        module = nn.Sigmoid()
        oracle_func = torch.sigmoid
        value_range = (0.0, 1.0)  # Sigmoid output range
    elif activation == "Tanh":
        module = nn.Tanh()
        oracle_func = torch.tanh
        value_range = (-1.0, 1.0)  # Tanh output range
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
    
    # 4. Value range check
    min_val, max_val = value_range
    assert torch.all(output >= min_val), f"Output values below minimum {min_val}"
    assert torch.all(output <= max_val), f"Output values above maximum {max_val}"
    
    # 5. Compare with oracle
    expected = oracle_func(x)
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-8), \
        f"Output doesn't match {activation} oracle"
    
    # 6. Monotonicity check (for normal input ranges)
    if input_range == "normal":
        # Create a simple test for monotonicity
        # For sigmoid: f'(x) = f(x)(1-f(x)) > 0 for all finite x
        # For tanh: f'(x) = 1 - tanh²(x) > 0 for all finite x
        # We'll test by checking that sorted inputs produce sorted outputs
        x_flat = x.flatten()
        sorted_x, indices = torch.sort(x_flat)
        output_flat = output.flatten()
        sorted_output = output_flat[indices]
        
        # Check if outputs are sorted (allow small numerical errors)
        diffs = sorted_output[1:] - sorted_output[:-1]
        # Allow small negative differences due to numerical precision
        assert torch.all(diffs >= -1e-7), f"{activation} fails monotonicity test"
    
    # 7. Special value checks for extreme inputs
    if input_range == "extreme" and activation == "Sigmoid":
        # For extreme values, sigmoid should approach 0 or 1
        # Check that very negative inputs produce near-zero outputs
        very_negative = x < -10.0
        if torch.any(very_negative):
            assert torch.all(output[very_negative] < 0.0001), \
                "Very negative inputs should produce near-zero sigmoid outputs"
        
        # Check that very positive inputs produce near-one outputs
        very_positive = x > 10.0
        if torch.any(very_positive):
            assert torch.all(output[very_positive] > 0.9999), \
                "Very positive inputs should produce near-one sigmoid outputs"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
# Test case: Softmax基础功能 (G2 group - placeholder)
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# Test case: Hardtanh阈值函数 (G3 group - placeholder)
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# Test case: LeakyReLU参数化测试 (deferred)
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
# Additional tests for G1 group activation functions

def test_relu_edge_cases():
    """Test ReLU with edge case inputs."""
    # Test with zeros
    x = torch.zeros(2, 3)
    relu = nn.ReLU()
    output = relu(x)
    assert torch.all(output == 0), "ReLU of zeros should be zeros"
    
    # Test with all negative values
    x = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    output = relu(x)
    assert torch.all(output == 0), "ReLU of negative values should be zeros"
    
    # Test with mixed values
    x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    output = relu(x)
    expected = torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 3.0]])
    assert torch.allclose(output, expected), "ReLU should zero out negative values"

def test_sigmoid_tanh_special_values():
    """Test Sigmoid and Tanh with special values."""
    # Test Sigmoid with zero
    sigmoid = nn.Sigmoid()
    x_zero = torch.zeros(1)
    output_sigmoid = sigmoid(x_zero)
    assert torch.allclose(output_sigmoid, torch.tensor([0.5]), rtol=1e-7), \
        "Sigmoid(0) should be 0.5"
    
    # Test Tanh with zero
    tanh = nn.Tanh()
    output_tanh = tanh(x_zero)
    assert torch.allclose(output_tanh, torch.tensor([0.0]), rtol=1e-7), \
        "Tanh(0) should be 0.0"
    
    # Test with large positive value
    x_large = torch.tensor([10.0])
    output_sigmoid_large = sigmoid(x_large)
    assert output_sigmoid_large > 0.999, "Sigmoid(10) should be > 0.999"
    
    output_tanh_large = tanh(x_large)
    assert output_tanh_large > 0.999, "Tanh(10) should be > 0.999"

def test_module_attributes():
    """Test that activation modules have correct attributes."""
    # Test ReLU
    relu = nn.ReLU(inplace=False)
    assert hasattr(relu, 'inplace'), "ReLU should have 'inplace' attribute"
    assert relu.inplace == False, "ReLU inplace should be False by default"
    
    # Test ReLU with inplace=True
    relu_inplace = nn.ReLU(inplace=True)
    assert relu_inplace.inplace == True, "ReLU inplace should be True when set"
    
    # Test Sigmoid (no special parameters)
    sigmoid = nn.Sigmoid()
    # Sigmoid has no special constructor parameters
    
    # Test Tanh (no special parameters)
    tanh = nn.Tanh()
    # Tanh has no special constructor parameters

def test_module_str_representation():
    """Test string representation of activation modules."""
    relu = nn.ReLU()
    assert "ReLU" in str(relu), "ReLU string representation should contain 'ReLU'"
    
    sigmoid = nn.Sigmoid()
    assert "Sigmoid" in str(sigmoid), "Sigmoid string representation should contain 'Sigmoid'"
    
    tanh = nn.Tanh()
    assert "Tanh" in str(tanh), "Tanh string representation should contain 'Tanh'"

# Cleanup and teardown
def teardown_module():
    """Cleanup after all tests."""
    # Clear any cached tensors
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
# ==== BLOCK:FOOTER END ====