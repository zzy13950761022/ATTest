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

# ==== BLOCK:HEADER START ====
# G3组测试文件头
# 激活与归一化函数族测试
# ==== BLOCK:HEADER END ====

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
    """Test quantized ReLU activation function."""
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
    
    # Perform quantized ReLU operation
    output = qF.relu(
        input=input_tensor,
        inplace=inplace
    )
    
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
# G3组测试文件尾部

def test_relu_inplace_vs_outplace():
    """Compare inplace and outplace ReLU operations."""
    # Create two identical tensors
    input1 = create_quantized_tensor([2, 3, 4, 4], scale=1.0, zero_point=0)
    input2 = input1.clone()
    
    # Apply ReLU inplace
    output_inplace = qF.relu(input1, inplace=True)
    
    # Apply ReLU outplace
    output_outplace = qF.relu(input2, inplace=False)
    
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