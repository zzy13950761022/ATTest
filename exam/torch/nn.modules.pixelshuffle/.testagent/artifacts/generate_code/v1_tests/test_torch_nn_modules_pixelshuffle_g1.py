import math
import pytest
import torch
import torch.nn as nn
from torch.nn.modules.pixelshuffle import PixelShuffle, PixelUnshuffle

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G1 group
@pytest.fixture
def set_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    return 42

def create_test_tensor(shape, dtype=torch.float32, device='cpu'):
    """Create test tensor with deterministic values"""
    torch.manual_seed(42)
    if shape[0] == 0:  # Empty batch
        return torch.empty(shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)

def assert_tensor_shapes_equal(actual, expected, msg=""):
    """Assert tensor shapes match"""
    assert actual.shape == expected.shape, f"{msg} Shape mismatch: {actual.shape} != {expected.shape}"

def assert_tensor_dtype_preserved(input_tensor, output_tensor, msg=""):
    """Assert output dtype matches input dtype"""
    assert output_tensor.dtype == input_tensor.dtype, f"{msg} Dtype mismatch: {output_tensor.dtype} != {input_tensor.dtype}"

def assert_tensor_finite(output_tensor, msg=""):
    """Assert tensor contains only finite values"""
    assert torch.isfinite(output_tensor).all(), f"{msg} Tensor contains non-finite values"

def assert_input_output_equal(input_tensor, output_tensor, msg=""):
    """Assert input and output tensors are equal"""
    assert torch.allclose(
        input_tensor,
        output_tensor,
        rtol=1e-6,
        atol=1e-6
    ), f"{msg} Input and output tensors differ"
# ==== BLOCK:HEADER END ====

class TestPixelShuffleG1:
    """Test cases for PixelShuffle module (G1 group)"""
    
    # ==== BLOCK:CASE_01 START ====
    @pytest.mark.parametrize("upscale_factor,input_shape,dtype,device", [
        (2, [1, 16, 4, 4], torch.float32, 'cpu'),
    ])
    def test_pixelshuffle_basic_shape_transform_g1(self, upscale_factor, input_shape, dtype, device):
        """TC-01: PixelShuffle基本形状变换
        
        Test basic shape transformation of PixelShuffle.
        Input shape: (*, C × r², H, W) -> Output shape: (*, C, H × r, W × r)
        """
        # Create PixelShuffle module
        pixel_shuffle = PixelShuffle(upscale_factor)
        
        # Create input tensor
        input_tensor = create_test_tensor(input_shape, dtype=dtype, device=device)
        
        # Forward pass
        output_tensor = pixel_shuffle(input_tensor)
        
        # Calculate expected output shape
        batch_size = input_shape[0]
        input_channels = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        
        expected_channels = input_channels // (upscale_factor * upscale_factor)
        expected_height = height * upscale_factor
        expected_width = width * upscale_factor
        expected_shape = (batch_size, expected_channels, expected_height, expected_width)
        
        # Weak assertions
        # 1. Output shape matches expected
        assert_tensor_shapes_equal(
            output_tensor, 
            torch.empty(expected_shape, dtype=dtype, device=device),
            "Output shape mismatch"
        )
        
        # 2. Dtype preserved
        assert_tensor_dtype_preserved(
            input_tensor,
            output_tensor,
            "Dtype not preserved"
        )
        
        # 3. Finite values
        assert_tensor_finite(
            output_tensor,
            "Output contains non-finite values"
        )
        
        # Additional verification: manual calculation for specific case
        if upscale_factor == 2 and input_shape == [1, 16, 4, 4]:
            # For upscale_factor=2, input (1, 16, 4, 4) -> output (1, 4, 8, 8)
            assert output_tensor.shape == torch.Size([1, 4, 8, 8])
            
            # Verify the transformation logic
            # Create a test pattern where each channel has unique values
            test_input = torch.arange(16 * 4 * 4, dtype=dtype, device=device).reshape(1, 16, 4, 4).float()
            test_output = pixel_shuffle(test_input)
            
            # Check that output height and width are doubled
            assert test_output.shape[2] == 8
            assert test_output.shape[3] == 8
            
            # Check that values are rearranged correctly
            assert torch.allclose(test_output[0, 0, 0, 0], test_input[0, 0, 0, 0])
            assert torch.allclose(test_output[0, 0, 0, 1], test_input[0, 1, 0, 0])
            assert torch.allclose(test_output[0, 0, 1, 0], test_input[0, 2, 0, 0])
            assert torch.allclose(test_output[0, 0, 1, 1], test_input[0, 3, 0, 0])
    # ==== BLOCK:CASE_01 END ====
    
    # ==== BLOCK:CASE_02 START ====
    @pytest.mark.parametrize("upscale_factor,input_shape,dtype,device", [
        (1, [2, 4, 8, 8], torch.float32, 'cpu'),
    ])
    def test_pixelshuffle_scale_factor_boundary_g1(self, upscale_factor, input_shape, dtype, device):
        """TC-02: PixelShuffle缩放因子边界
        
        Test PixelShuffle with scale factor = 1 (identity transformation).
        When upscale_factor=1, output should be identical to input.
        """
        # Create PixelShuffle module with scale factor 1
        pixel_shuffle = PixelShuffle(upscale_factor)
        
        # Create input tensor
        input_tensor = create_test_tensor(input_shape, dtype=dtype, device=device)
        
        # Forward pass
        output_tensor = pixel_shuffle(input_tensor)
        
        # Weak assertions
        # 1. Output shape matches input shape (identity transformation)
        assert_tensor_shapes_equal(
            output_tensor,
            input_tensor,
            "Output shape should match input shape for upscale_factor=1"
        )
        
        # 2. Input and output should be equal (identity check)
        assert_input_output_equal(
            input_tensor,
            output_tensor,
            "Output should be identical to input for upscale_factor=1"
        )
        
        # 3. Finite values
        assert_tensor_finite(
            output_tensor,
            "Output contains non-finite values"
        )
        
        # Additional verification for identity property
        assert output_tensor.shape == input_tensor.shape
        
        # Verify all values are exactly the same
        assert torch.equal(output_tensor, input_tensor)
        
        # Test with different random inputs to ensure it's not a coincidence
        for _ in range(3):
            random_input = torch.randn_like(input_tensor)
            random_output = pixel_shuffle(random_input)
            assert torch.allclose(random_output, random_input, rtol=1e-6, atol=1e-6)
    # ==== BLOCK:CASE_02 END ====
    
    # ==== BLOCK:CASE_03 START ====
    # PixelShuffle不同批次大小 (DEFERRED - placeholder only)
    # TC-03: PixelShuffle不同批次大小
    # Parameters: upscale_factor=3, input_shape=[0, 27, 6, 6], dtype=float32, device=cpu
    # Assertions (weak): output_shape, dtype_preserved, finite_values
    # ==== BLOCK:CASE_03 END ====
    
    # ==== BLOCK:CASE_04 START ====
    # PixelShuffle不同数据类型 (DEFERRED - placeholder only)
    # TC-04: PixelShuffle不同数据类型
    # Parameters: upscale_factor=2, input_shape=[1, 16, 4, 4], dtype=float64, device=cpu
    # Assertions (weak): output_shape, dtype_preserved, finite_values
    # ==== BLOCK:CASE_04 END ====

# ==== BLOCK:FOOTER START ====
# Additional test cases and edge case tests for G1 group

def test_pixelshuffle_invalid_upscale_factor_g1():
    """Test PixelShuffle with invalid upscale factor (G1 group)"""
    with pytest.raises(ValueError):
        PixelShuffle(0)
    
    with pytest.raises(ValueError):
        PixelShuffle(-1)

def test_pixelshuffle_invalid_input_channels_g1():
    """Test PixelShuffle with input channels not divisible by r² (G1 group)"""
    pixel_shuffle = PixelShuffle(2)
    # Input channels = 15, not divisible by 4
    input_tensor = torch.randn(1, 15, 4, 4)
    with pytest.raises(RuntimeError):
        pixel_shuffle(input_tensor)

def test_pixelshuffle_minimum_dimensions_g1():
    """Test PixelShuffle with minimum required dimensions (G1 group)"""
    pixel_shuffle = PixelShuffle(2)
    # 4D tensor (minimum)
    input_tensor = torch.randn(1, 4, 2, 2)
    output = pixel_shuffle(input_tensor)
    assert output.dim() == 4
    
    # 5D tensor (batch + 4D)
    input_tensor_5d = torch.randn(2, 1, 4, 4, 4)
    output_5d = pixel_shuffle(input_tensor_5d)
    assert output_5d.dim() == 5

def test_pixelshuffle_different_upscale_factors_g1():
    """Test PixelShuffle with different upscale factors (G1 group)"""
    # Test various upscale factors
    test_cases = [
        (2, [1, 16, 4, 4], [1, 4, 8, 8]),
        (3, [1, 27, 3, 3], [1, 3, 9, 9]),
        (4, [1, 64, 2, 2], [1, 4, 8, 8]),
    ]
    
    for upscale_factor, input_shape, expected_shape in test_cases:
        pixel_shuffle = PixelShuffle(upscale_factor)
        input_tensor = torch.randn(input_shape)
        output = pixel_shuffle(input_tensor)
        
        assert output.shape == torch.Size(expected_shape)
        assert output.dtype == input_tensor.dtype
        assert torch.isfinite(output).all()

def test_pixelshuffle_gradient_flow_g1():
    """Test that gradients can flow through PixelShuffle (G1 group)"""
    pixel_shuffle = PixelShuffle(2)
    
    # Create input with requires_grad
    input_tensor = torch.randn(1, 16, 4, 4, requires_grad=True)
    
    # Forward pass
    output = pixel_shuffle(input_tensor)
    
    # Create a loss and backward pass
    loss = output.sum()
    loss.backward()
    
    # Check that gradients were computed
    assert input_tensor.grad is not None
    assert input_tensor.grad.shape == input_tensor.shape
    assert torch.isfinite(input_tensor.grad).all()

# ==== BLOCK:FOOTER END ====