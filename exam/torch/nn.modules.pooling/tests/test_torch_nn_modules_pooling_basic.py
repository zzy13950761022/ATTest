import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

# ==== BLOCK:HEADER START ====
# Test file for torch.nn.modules.pooling - Basic pooling functions (G1)
# This file contains tests for MaxPool1d, AvgPool1d, MaxPool2d, AvgPool2d
# 
# Test structure:
# 1. Basic instantiation and forward pass
# 2. Output shape calculation
# 3. Parameter validation
# 4. Edge cases and boundary conditions
# ==== BLOCK:HEADER END ====

# Helper functions and fixtures
@pytest.fixture
def set_random_seed():
    """Set random seed for reproducible tests"""
    torch.manual_seed(42)
    return 42

@pytest.fixture
def create_test_tensor():
    """Create test tensor with given shape and dtype"""
    def _create(shape, dtype=torch.float32, device='cpu'):
        tensor = torch.randn(*shape, dtype=dtype, device=device)
        # Scale to reasonable range
        tensor = tensor * 2.0 - 1.0  # Range [-1, 1]
        return tensor
    return _create

# Test class for basic pooling functions
class TestBasicPooling:
    """Test cases for basic pooling layers (MaxPool1d, AvgPool1d, MaxPool2d, AvgPool2d)"""
    
    def test_maxpool1d_basic(self, set_random_seed, create_test_tensor):
        """TC-01: MaxPool1d基本功能
        
        Test basic functionality of MaxPool1d with standard parameters.
        Weak asserts: instance_created, forward_works, output_shape_correct, output_dtype_preserved
        """
        # Test parameters from test_plan.json
        kernel_size = 3
        stride = 2
        padding = 1
        dilation = 1
        return_indices = False
        ceil_mode = False
        input_shape = (2, 3, 10)  # (batch, channels, length)
        dtype = torch.float32
        device = 'cpu'
        
        # Create test input tensor
        x = create_test_tensor(input_shape, dtype=dtype, device=device)
        
        # 1. Test instance creation
        pool_layer = nn.MaxPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )
        
        # Assert: instance_created
        assert pool_layer is not None
        assert isinstance(pool_layer, nn.MaxPool1d)
        
        # Check layer parameters
        assert pool_layer.kernel_size == kernel_size
        assert pool_layer.stride == stride
        assert pool_layer.padding == padding
        assert pool_layer.dilation == dilation
        assert pool_layer.return_indices == return_indices
        assert pool_layer.ceil_mode == ceil_mode
        
        # 2. Test forward pass
        output = pool_layer(x)
        
        # Assert: forward_works
        assert output is not None
        assert isinstance(output, torch.Tensor)
        
        # 3. Assert: output_shape_correct
        # Calculate expected output shape using the formula from PyTorch docs:
        # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        L_in = input_shape[2]
        L_out = math.floor(
            (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
        expected_shape = (input_shape[0], input_shape[1], L_out)
        
        assert output.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {output.shape}"
        
        # 4. Assert: output_dtype_preserved
        assert output.dtype == dtype, \
            f"Expected dtype {dtype}, got {output.dtype}"
        
        # 5. Additional weak assertion: output values are reasonable
        # Max pooling should output values from the input tensor
        assert torch.all(output >= x.min()), "Output contains values smaller than input min"
        assert torch.all(output <= x.max()), "Output contains values larger than input max"
        
        # 6. Verify against functional implementation (weak oracle check)
        # Using torch.nn.functional.max_pool1d as oracle
        expected_output = F.max_pool1d(
            x, kernel_size, stride, padding, dilation, 
            ceil_mode=ceil_mode, return_indices=return_indices
        )
        
        # Check if outputs match (allowing for floating point differences)
        assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-8), \
            "Output doesn't match functional implementation"
    
    def test_avgpool2d_basic(self, set_random_seed, create_test_tensor):
        """TC-02: AvgPool2d基本功能
        
        Test basic functionality of AvgPool2d with standard parameters.
        Weak asserts: instance_created, forward_works, output_shape_correct, output_dtype_preserved
        """
        # Test parameters from test_plan.json
        kernel_size = (2, 2)
        stride = (2, 2)
        padding = 0
        dilation = 1
        ceil_mode = False
        input_shape = (4, 3, 8, 8)  # (batch, channels, height, width)
        dtype = torch.float32
        device = 'cpu'
        
        # Create test input tensor
        x = create_test_tensor(input_shape, dtype=dtype, device=device)
        
        # 1. Test instance creation
        # Note: AvgPool2d does NOT support dilation parameter
        pool_layer = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode
        )
        
        # Assert: instance_created
        assert pool_layer is not None
        assert isinstance(pool_layer, nn.AvgPool2d)
        
        # Check layer parameters
        assert pool_layer.kernel_size == kernel_size
        assert pool_layer.stride == stride
        assert pool_layer.padding == padding
        assert pool_layer.ceil_mode == ceil_mode
        
        # 2. Test forward pass
        output = pool_layer(x)
        
        # Assert: forward_works
        assert output is not None
        assert isinstance(output, torch.Tensor)
        
        # 3. Assert: output_shape_correct
        # Calculate expected output shape using the formula from PyTorch docs:
        # H_out = floor((H_in + 2*padding - (kernel_size[0]-1) - 1)/stride[0] + 1)
        # W_out = floor((W_in + 2*padding - (kernel_size[1]-1) - 1)/stride[1] + 1)
        # Note: AvgPool2d does not have dilation, so dilation is effectively 1
        H_in, W_in = input_shape[2], input_shape[3]
        H_out = math.floor(
            (H_in + 2 * padding - (kernel_size[0] - 1) - 1) / stride[0] + 1
        )
        W_out = math.floor(
            (W_in + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1
        )
        expected_shape = (input_shape[0], input_shape[1], H_out, W_out)
        
        assert output.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {output.shape}"
        
        # 4. Assert: output_dtype_preserved
        assert output.dtype == dtype, \
            f"Expected dtype {dtype}, got {output.dtype}"
        
        # 5. Additional weak assertion: output values are reasonable
        # Average pooling should output values within the range of input values
        # Since we're averaging, the output should be between min and max of each pooling window
        # For a simple check, verify it's not extreme
        assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
        
        # 6. Verify against functional implementation (weak oracle check)
        # Using torch.nn.functional.avg_pool2d as oracle
        expected_output = F.avg_pool2d(
            x, kernel_size, stride, padding, 
            ceil_mode=ceil_mode
        )
        
        # Check if outputs match (allowing for floating point differences)
        assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-8), \
            "Output doesn't match functional implementation"
        
        # 7. Test that average pooling computes correct averages
        # For a simple 2x2 average pool with stride 2, each output element should be
        # the average of the corresponding 2x2 block in the input
        # Test this for a few positions
        batch_idx, channel_idx = 0, 0
        for i in range(min(2, H_out)):
            for j in range(min(2, W_out)):
                input_block = x[batch_idx, channel_idx, 
                               i*stride[0]:i*stride[0]+kernel_size[0],
                               j*stride[1]:j*stride[1]+kernel_size[1]]
                expected_avg = torch.mean(input_block)
                actual_avg = output[batch_idx, channel_idx, i, j]
                assert torch.allclose(actual_avg, expected_avg, rtol=1e-5, atol=1e-8), \
                    f"Average pooling incorrect at position ({i},{j})"
    
    # ==== BLOCK:CASE_03 START ====
    # Placeholder for MaxPool3d with return_indices (DEFERRED)
    # TC-03: MaxPool3d与return_indices
    # This test case is deferred to later rounds
    # ==== BLOCK:CASE_03 END ====
    
    # ==== BLOCK:CASE_04 START ====
    # Placeholder for pooling parameter boundary validation (DEFERRED)
    # TC-04: 池化参数边界验证
    # This test case is deferred to later rounds
    # ==== BLOCK:CASE_04 END ====

# ==== BLOCK:FOOTER START ====
# Footer block for test file
# Contains additional test cases and cleanup if needed
# ==== BLOCK:FOOTER END ====