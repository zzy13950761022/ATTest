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
# G2组测试文件头
# 线性与池化函数族测试
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: linear基本量化操作
@pytest.mark.parametrize("test_params", [
    {
        "input_shape": [2, 4],
        "weight_shape": [3, 4],
        "input_dtype": torch.quint8,
        "weight_dtype": torch.qint8,
        "bias": True,
        "scale": 1.0,
        "zero_point": 0
    }
])
def test_linear_basic_quantized_operation(test_params):
    """Test basic quantized linear operation."""
    # Unpack parameters
    input_shape = test_params["input_shape"]
    weight_shape = test_params["weight_shape"]
    input_dtype = test_params["input_dtype"]
    weight_dtype = test_params["weight_dtype"]
    bias = test_params["bias"]
    scale = test_params["scale"]
    zero_point = test_params["zero_point"]
    
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
        # For linear operation, bias is float
        bias_tensor = torch.randn(weight_shape[0])
    
    # Calculate expected output shape
    # Linear: input [batch, in_features] * weight [out_features, in_features]^T
    # -> output [batch, out_features]
    expected_shape = (input_shape[0], weight_shape[0])
    
    # Perform quantized linear operation
    output = qF.linear(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
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
    
    # Verify that the operation produces reasonable values
    # Dequantize and check range
    dequantized_output = output.dequantize()
    assert torch.all(torch.isfinite(dequantized_output)), \
        "Dequantized output should contain only finite values"
    
    # Check that output values are within reasonable range
    # For scale=1.0, zero_point=0, values should be in typical activation range
    output_mean = dequantized_output.mean().item()
    output_std = dequantized_output.std().item()
    assert abs(output_mean) < 10.0, f"Output mean {output_mean} seems too large"
    assert output_std > 0.01, f"Output std {output_std} seems too small"
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# TC-06: avg_pool2d量化操作 (DEFERRED)
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# TC-07: max_pool2d量化操作 (DEFERRED)
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:FOOTER START ====
# G2组测试文件尾部

def test_quantized_linear_weight_packing():
    """Test that linear operation handles weight packing correctly."""
    # Create quantized input
    input_tensor = create_quantized_tensor([2, 4], scale=1.0, zero_point=0)
    
    # Create quantized weight
    weight_tensor = create_quantized_weight([3, 4], scale=1.0, zero_point=0)
    
    # Perform linear operation
    output = qF.linear(input_tensor, weight_tensor)
    
    # Basic assertions
    assert output.is_quantized
    assert output.shape == (2, 3)
    assert output.dtype == torch.quint8

# Cleanup and teardown if needed
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Clear any cached data if needed
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
# ==== BLOCK:FOOTER END ====