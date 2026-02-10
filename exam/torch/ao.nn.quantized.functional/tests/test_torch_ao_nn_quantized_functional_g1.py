import math
import pytest
import torch
import torch.nn.functional as F
from torch.ao.nn.quantized import functional as qF

# Set random seed for reproducibility
torch.manual_seed(42)

def create_quantized_tensor(shape, dtype=torch.quint8, scale=1.0, zero_point=0):
    """Create a quantized tensor with random data."""
    # Generate random float data
    float_data = torch.randn(shape)
    
    # Quantize the data
    if dtype == torch.quint8:
        # For quint8, clamp to [0, 255] range
        float_data = float_data.clamp(-2, 2)  # Keep in reasonable range
        q_data = torch.quantize_per_tensor(
            float_data, scale=scale, zero_point=zero_point, dtype=dtype
        )
    elif dtype == torch.qint8:
        # For qint8, clamp to [-128, 127] range
        float_data = float_data.clamp(-2, 2)
        q_data = torch.quantize_per_tensor(
            float_data, scale=scale, zero_point=zero_point, dtype=dtype
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    return q_data

def create_quantized_weight(shape, dtype=torch.qint8, scale=1.0, zero_point=0):
    """Create a quantized weight tensor."""
    return create_quantized_tensor(shape, dtype, scale, zero_point)

def create_quantized_bias(out_channels, weight_scale, input_scale):
    """Create a float bias tensor for quantized operations."""
    # Bias in quantized operations is typically float
    bias = torch.randn(out_channels)
    # Scale bias appropriately for quantized operations
    bias = bias * (weight_scale * input_scale)
    return bias

def assert_quantized_tensor_properties(tensor, expected_shape=None, 
                                      expected_dtype=None, expected_scale=None,
                                      expected_zero_point=None):
    """Assert that a tensor has the expected quantized properties."""
    assert tensor.is_quantized, "Tensor should be quantized"
    
    if expected_shape is not None:
        assert tensor.shape == torch.Size(expected_shape), \
            f"Expected shape {expected_shape}, got {tensor.shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, \
            f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    if expected_scale is not None:
        assert math.isclose(tensor.q_scale(), expected_scale, rel_tol=1e-6), \
            f"Expected scale {expected_scale}, got {tensor.q_scale()}"
    
    if expected_zero_point is not None:
        assert tensor.q_zero_point() == expected_zero_point, \
            f"Expected zero_point {expected_zero_point}, got {tensor.q_zero_point()}"
    
    # Check for NaN or Inf values
    assert not torch.any(torch.isnan(tensor.dequantize())), "Tensor contains NaN values"
    assert not torch.any(torch.isinf(tensor.dequantize())), "Tensor contains Inf values"

def calculate_conv_output_shape(input_shape, weight_shape, stride=1, padding=0, dilation=1):
    """Calculate output shape for convolution operation."""
    N, C_in, *spatial_dims = input_shape
    C_out, C_in_div_groups, *kernel_dims = weight_shape
    
    output_dims = []
    for i, (input_dim, kernel_dim) in enumerate(zip(spatial_dims, kernel_dims)):
        output_dim = math.floor(
            (input_dim + 2 * padding - dilation * (kernel_dim - 1) - 1) / stride + 1
        )
        output_dims.append(output_dim)
    
    return (N, C_out, *output_dims)

# ==== BLOCK:HEADER START ====
# G1组测试文件头
# 核心卷积函数族测试
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# TC-01: conv2d基本量化操作
@pytest.mark.parametrize("test_params", [
    {
        "input_shape": [1, 3, 5, 5],
        "weight_shape": [2, 3, 3, 3],
        "input_dtype": torch.quint8,
        "weight_dtype": torch.qint8,
        "bias": True,
        "scale": 1.0,
        "zero_point": 0,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1
    }
])
def test_conv2d_basic_quantized_operation(test_params):
    """Test basic quantized conv2d operation."""
    # Initialize quantization engine
    torch.backends.quantized.engine = 'qnnpack'  # or 'fbgemm'
    
    # Unpack parameters
    input_shape = test_params["input_shape"]
    weight_shape = test_params["weight_shape"]
    input_dtype = test_params["input_dtype"]
    weight_dtype = test_params["weight_dtype"]
    bias = test_params["bias"]
    scale = test_params["scale"]
    zero_point = test_params["zero_point"]
    stride = test_params["stride"]
    padding = test_params["padding"]
    dilation = test_params["dilation"]
    groups = test_params["groups"]
    
    # Create quantized input tensor
    input_tensor = create_quantized_tensor(
        input_shape, dtype=input_dtype, scale=scale, zero_point=zero_point
    )
    
    # Create quantized weight tensor
    weight_tensor = create_quantized_weight(
        weight_shape, dtype=weight_dtype, scale=scale, zero_point=0
    )
    
    # Create bias if needed
    bias_tensor = None
    if bias:
        bias_tensor = create_quantized_bias(
            weight_shape[0], weight_scale=scale, input_scale=scale
        )
    
    # Calculate expected output shape
    expected_shape = calculate_conv_output_shape(
        input_shape, weight_shape, stride=stride, padding=padding, dilation=dilation
    )
    
    # Perform quantized convolution
    output = qF.conv2d(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        scale=scale,
        zero_point=zero_point
    )
    
    # Weak assertions (first round)
    # 1. Output is quantized
    assert output.is_quantized, "Output should be quantized"
    
    # 2. Output shape is correct
    assert output.shape == torch.Size(expected_shape), \
        f"Expected shape {expected_shape}, got {output.shape}"
    
    # 3. Output dtype is correct
    assert output.dtype == input_dtype, \
        f"Expected dtype {input_dtype}, got {output.dtype}"
    
    # 4. No NaN or Inf values
    assert not torch.any(torch.isnan(output.dequantize())), "Output contains NaN values"
    assert not torch.any(torch.isinf(output.dequantize())), "Output contains Inf values"
    
    # Additional basic checks
    assert output.q_scale() == scale, \
        f"Expected scale {scale}, got {output.q_scale()}"
    assert output.q_zero_point() == zero_point, \
        f"Expected zero_point {zero_point}, got {output.q_zero_point()}"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# TC-02: conv2d量化参数传播
@pytest.mark.parametrize("test_params", [
    {
        "input_shape": [2, 4, 6, 6],
        "weight_shape": [4, 4, 2, 2],
        "input_dtype": torch.quint8,
        "weight_dtype": torch.qint8,
        "bias": False,
        "scale": 0.5,
        "zero_point": 128,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "groups": 1
    }
])
def test_conv2d_quantization_parameter_propagation(test_params):
    """Test that quantization parameters are correctly propagated in conv2d."""
    # Initialize quantization engine
    torch.backends.quantized.engine = 'qnnpack'  # or 'fbgemm'
    
    # Unpack parameters
    input_shape = test_params["input_shape"]
    weight_shape = test_params["weight_shape"]
    input_dtype = test_params["input_dtype"]
    weight_dtype = test_params["weight_dtype"]
    bias = test_params["bias"]
    scale = test_params["scale"]
    zero_point = test_params["zero_point"]
    stride = test_params["stride"]
    padding = test_params["padding"]
    dilation = test_params["dilation"]
    groups = test_params["groups"]
    
    # Create quantized input tensor with specific scale and zero_point
    input_tensor = create_quantized_tensor(
        input_shape, dtype=input_dtype, scale=scale, zero_point=zero_point
    )
    
    # Create quantized weight tensor (weight typically has zero_point=0 for qint8)
    weight_tensor = create_quantized_weight(
        weight_shape, dtype=weight_dtype, scale=scale, zero_point=0
    )
    
    # Create bias if needed
    bias_tensor = None
    if bias:
        bias_tensor = create_quantized_bias(
            weight_shape[0], weight_scale=scale, input_scale=scale
        )
    
    # Calculate expected output shape
    expected_shape = calculate_conv_output_shape(
        input_shape, weight_shape, stride=stride, padding=padding, dilation=dilation
    )
    
    # Perform quantized convolution
    output = qF.conv2d(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        scale=scale,
        zero_point=zero_point
    )
    
    # Weak assertions (first round)
    # 1. Output is quantized
    assert output.is_quantized, "Output should be quantized"
    
    # 2. Output shape is correct
    assert output.shape == torch.Size(expected_shape), \
        f"Expected shape {expected_shape}, got {output.shape}"
    
    # 3. Scale and zero_point match the specified values
    assert math.isclose(output.q_scale(), scale, rel_tol=1e-6), \
        f"Expected scale {scale}, got {output.q_scale()}"
    assert output.q_zero_point() == zero_point, \
        f"Expected zero_point {zero_point}, got {output.q_zero_point()}"
    
    # 4. No NaN or Inf values
    assert not torch.any(torch.isnan(output.dequantize())), "Output contains NaN values"
    assert not torch.any(torch.isinf(output.dequantize())), "Output contains Inf values"
    
    # Additional checks for parameter propagation
    # Verify that stride and padding affect output shape correctly
    # For stride=2, padding=1, input 6x6 with kernel 2x2 should output 4x4
    N, C_out, H_out, W_out = output.shape
    assert H_out == 4, f"Expected height 4 with stride=2, padding=1, got {H_out}"
    assert W_out == 4, f"Expected width 4 with stride=2, padding=1, got {W_out}"
    
    # Verify output dtype matches input dtype
    assert output.dtype == input_dtype, \
        f"Output dtype {output.dtype} should match input dtype {input_dtype}"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: conv1d基本功能 (DEFERRED)
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: conv3d基本功能 (DEFERRED)
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:FOOTER START ====
# G1组测试文件尾部

def test_quantized_input_validation():
    """Test that non-quantized inputs raise ValueError."""
    # Create a regular (non-quantized) tensor
    regular_tensor = torch.randn(1, 3, 5, 5)
    
    # Try to use it with quantized conv2d - should raise ValueError
    with pytest.raises(ValueError, match="must be quantized"):
        # Create a quantized weight tensor
        weight = create_quantized_weight([2, 3, 3, 3])
        # conv2d requires bias as third positional argument, even if None
        qF.conv2d(regular_tensor, weight, None)

def test_conv2d_with_different_quantization_params():
    """Test conv2d with different input and output quantization parameters."""
    # Create input with one set of parameters
    input_tensor = create_quantized_tensor(
        [1, 3, 5, 5], scale=0.5, zero_point=64
    )
    
    # Create weight
    weight_tensor = create_quantized_weight([2, 3, 3, 3], scale=0.5, zero_point=0)
    
    # Specify different output quantization parameters
    output_scale = 0.25
    output_zero_point = 128
    
    # conv2d requires bias as third positional argument
    output = qF.conv2d(
        input_tensor, weight_tensor, None,  # bias=None
        scale=output_scale, zero_point=output_zero_point
    )
    
    # Check that output uses the specified parameters
    assert output.is_quantized
    assert math.isclose(output.q_scale(), output_scale, rel_tol=1e-6)
    assert output.q_zero_point() == output_zero_point

# Cleanup and teardown if needed
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Clear any cached data if needed
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
# ==== BLOCK:FOOTER END ====