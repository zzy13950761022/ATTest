import math
import pytest
import torch
import torch.nn.functional as F
from torch.ao.nn.quantized import functional as qF

# ==== BLOCK:HEADER START ====
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

# ==== BLOCK:CASE_08 START ====
# TC-08: relu量化激活
@pytest.mark.parametrize("test_params", [
    {
        "input_shape": [2, 3, 4, 4],
        "input_dtype": torch.quint8,
        "scale": 1.0,
        "zero_point": 0,
        "inplace": False
    }
])
def test_relu_quantized_activation(test_params):
    """Test quantized ReLU activation function using clamp."""
    # Unpack parameters
    input_shape = test_params["input_shape"]
    input_dtype = test_params["input_dtype"]
    scale = test_params["scale"]
    zero_point = test_params["zero_point"]
    inplace = test_params["inplace"]
    
    # Create quantized input tensor with both positive and negative values
    # First create float data with both positive and negative values
    float_data = torch.randn(input_shape)
    # Ensure we have both positive and negative values for ReLU test
    float_data = float_data * 2.0  # Scale to get wider range
    
    # Quantize the data
    input_tensor = torch.quantize_per_tensor(
        float_data, scale=scale, zero_point=zero_point, dtype=input_dtype
    )
    
    # Store original tensor for comparison if not inplace
    original_tensor = input_tensor.clone() if not inplace else None
    
    # Perform quantized ReLU operation using clamp
    # Note: In quantized functional, ReLU is implemented via clamp with min=0
    # There's no inplace parameter for clamp, so we need to handle it differently
    if inplace:
        # For inplace operation, we need to modify the input tensor
        # Since clamp doesn't have inplace parameter, we'll mark this test as xfail
        # because torch.clamp_ doesn't support quantized tensors
        pytest.xfail("torch.clamp_ doesn't support quantized tensors for inplace operation")
        output = torch.clamp_(input_tensor, min=0)
    else:
        # For outplace operation, use qF.clamp
        output = qF.clamp(input_tensor, min_=0, max_=float('inf'))
    
    # Weak assertions (first round)
    # 1. Output is quantized
    assert output.is_quantized, "Output should be quantized"
    
    # 2. Output shape is correct (same as input)
    assert output.shape == torch.Size(input_shape), \
        f"Expected shape {input_shape}, got {output.shape}"
    
    # 3. Output dtype is correct
    assert output.dtype == input_dtype, \
        f"Expected dtype {input_dtype}, got {output.dtype}"
    
    # 4. No NaN or Inf values
    assert not torch.any(torch.isnan(output.dequantize())), "Output contains NaN values"
    assert not torch.any(torch.isinf(output.dequantize())), "Output contains Inf values"
    
    # 5. ReLU effect is visible (all values >= 0 for zero_point=0)
    dequantized_output = output.dequantize()
    if zero_point == 0:
        # For zero_point=0, quantized values >= 0 correspond to dequantized values >= 0
        # Check that all values are non-negative (allow small numerical errors)
        assert torch.all(dequantized_output >= -1e-6), \
            "ReLU should produce non-negative output"
    
    # Additional checks for ReLU properties
    # Check that quantization parameters are preserved
    assert output.q_scale() == scale, \
        f"Expected scale {scale}, got {output.q_scale()}"
    assert output.q_zero_point() == zero_point, \
        f"Expected zero_point {zero_point}, got {output.q_zero_point()}"
    
    # Check that negative values are zeroed (for zero_point=0)
    if zero_point == 0:
        dequantized_input = input_tensor.dequantize()
        dequantized_output = output.dequantize()
        
        # For each position, output should be max(0, input)
        for i in range(dequantized_input.numel()):
            input_val = dequantized_input.view(-1)[i].item()
            output_val = dequantized_output.view(-1)[i].item()
            expected_val = max(0.0, input_val)
            
            # Allow small numerical errors in quantization
            assert abs(output_val - expected_val) < 1e-3, \
                f"ReLU failed at position {i}: input={input_val}, output={output_val}, expected={expected_val}"
    
    # If not inplace, check that input is unchanged
    if not inplace and original_tensor is not None:
        # Compare dequantized values
        input_dequantized = input_tensor.dequantize()
        original_dequantized = original_tensor.dequantize()
        assert torch.allclose(input_dequantized, original_dequantized, rtol=1e-6), \
            "Input tensor was modified in non-inplace operation"
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:CASE_09 START ====
# TC-09: hardtanh量化激活 (DEFERRED)
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:CASE_10 START ====
# TC-10: layer_norm量化归一化 (DEFERRED)
# ==== BLOCK:CASE_10 END ====

# ==== BLOCK:FOOTER START ====
# Additional tests and cleanup

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

@pytest.mark.xfail(reason="qF.relu不存在，量化ReLU应使用qF.clamp实现")
def test_relu_inplace_vs_outplace():
    """Compare inplace and outplace ReLU operations."""
    # Create two identical tensors
    input1 = create_quantized_tensor([2, 3, 4, 4], scale=1.0, zero_point=0)
    input2 = input1.clone()
    
    # Apply ReLU inplace using clamp (note: clamp doesn't have inplace parameter)
    # This will fail because torch.clamp_ doesn't support quantized tensors
    output_inplace = torch.clamp_(input1, min=0)
    
    # Apply ReLU outplace using qF.clamp
    output_outplace = qF.clamp(input2, min_=0, max_=float('inf'))
    
    # Results should be the same
    assert torch.allclose(
        output_inplace.dequantize(),
        output_outplace.dequantize(),
        rtol=1e-6
    )
    
    # For inplace, input1 should be the same object as output_inplace
    assert input1 is output_inplace
    
    # For outplace, input2 should be different from output_outplace
    assert input2 is not output_outplace

# Cleanup and teardown if needed
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Clear any cached data if needed
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
# ==== BLOCK:FOOTER END ====