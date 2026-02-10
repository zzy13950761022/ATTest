import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

# ==== BLOCK:HEADER START ====
# Test file for torch.nn.modules.pooling - Advanced pooling functions (G2)
# This file contains tests for AdaptiveMaxPool1d, AdaptiveAvgPool1d, AdaptiveMaxPool2d, AdaptiveAvgPool2d
# 
# Test structure:
# 1. Adaptive pooling with fixed output size
# 2. Return indices for adaptive max pooling
# 3. Parameter validation for adaptive pooling
# 4. Edge cases and boundary conditions
# ==== BLOCK:HEADER END ====

# Helper functions and fixtures (same as basic file)
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

# Test class for advanced pooling functions
class TestAdvancedPooling:
    """Test cases for advanced pooling layers (AdaptiveMaxPool1d, AdaptiveAvgPool1d, AdaptiveMaxPool2d, AdaptiveAvgPool2d)"""
    
    def test_adaptive_avgpool2d_basic(self, set_random_seed, create_test_tensor):
        """TC-05: AdaptiveAvgPool2d基本功能
        
        Test basic functionality of AdaptiveAvgPool2d with fixed output size.
        Weak asserts: instance_created, forward_works, output_shape_matches_output_size, output_dtype_preserved
        """
        # Test parameters from test_plan.json
        output_size = (4, 4)
        input_shape = (2, 3, 8, 8)  # (batch, channels, height, width)
        dtype = torch.float32
        device = 'cpu'
        
        # Create test input tensor
        x = create_test_tensor(input_shape, dtype=dtype, device=device)
        
        # 1. Test instance creation
        pool_layer = nn.AdaptiveAvgPool2d(output_size=output_size)
        
        # Assert: instance_created
        assert pool_layer is not None
        assert isinstance(pool_layer, nn.AdaptiveAvgPool2d)
        
        # Check layer parameters
        assert pool_layer.output_size == output_size
        
        # 2. Test forward pass
        output = pool_layer(x)
        
        # Assert: forward_works
        assert output is not None
        assert isinstance(output, torch.Tensor)
        
        # 3. Assert: output_shape_matches_output_size
        # Adaptive pooling should produce exactly the specified output size
        expected_shape = (input_shape[0], input_shape[1], output_size[0], output_size[1])
        assert output.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {output.shape}"
        
        # 4. Assert: output_dtype_preserved
        assert output.dtype == dtype, \
            f"Expected dtype {dtype}, got {output.dtype}"
        
        # 5. Additional weak assertion: output values are reasonable
        # Adaptive average pooling should output values within the range of input values
        assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
        
        # 6. Verify against functional implementation (weak oracle check)
        # Using torch.nn.functional.adaptive_avg_pool2d as oracle
        expected_output = F.adaptive_avg_pool2d(x, output_size)
        
        # Check if outputs match (allowing for floating point differences)
        assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-8), \
            "Output doesn't match functional implementation"
        
        # 7. Test adaptive behavior: output size should be exactly as specified
        # regardless of input size (within reasonable bounds)
        assert output.size(2) == output_size[0], \
            f"Output height should be {output_size[0]}, got {output.size(2)}"
        assert output.size(3) == output_size[1], \
            f"Output width should be {output_size[1]}, got {output.size(3)}"
    
    # ==== BLOCK:CASE_06 START ====
    # Placeholder for AdaptiveMaxPool1d with return_indices (DEFERRED)
    # TC-06: AdaptiveMaxPool1d与return_indices
    # This test case is deferred to later rounds
    # ==== BLOCK:CASE_06 END ====
    
    # ==== BLOCK:CASE_07 START ====
    # Placeholder for pooling parameter exception validation (DEFERRED)
    # TC-07: 池化参数异常验证
    # This test case is deferred to later rounds
    # ==== BLOCK:CASE_07 END ====

# ==== BLOCK:FOOTER START ====
# Footer block for test file
# Contains additional test cases and cleanup if needed
# ==== BLOCK:FOOTER END ====