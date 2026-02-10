import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

# ==== BLOCK:HEADER START ====
# Test file for torch.nn.modules.pooling - Special pooling functions (G3)
# This file contains tests for FractionalMaxPool2d, LPPool1d, MaxUnpool1d
# 
# Test structure:
# 1. Fractional max pooling with random regions
# 2. LP pooling with different norm types
# 3. Max unpooling reconstruction
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

# Test class for special pooling functions
class TestSpecialPooling:
    """Test cases for special pooling layers (FractionalMaxPool2d, LPPool1d, MaxUnpool1d)"""
    
    @patch('torch.nn.functional.fractional_max_pool2d')
    def test_fractional_maxpool2d_basic(self, mock_fractional_pool, set_random_seed, create_test_tensor):
        """TC-08: FractionalMaxPool2d基本功能
        
        Test basic functionality of FractionalMaxPool2d with fixed output size.
        Weak asserts: instance_created, forward_works, output_shape_correct, output_dtype_preserved
        Requires mock: True (for random number generator)
        """
        # Test parameters from test_plan.json
        kernel_size = 2
        output_size = (3, 3)
        output_ratio = None
        return_indices = False
        input_shape = (2, 3, 8, 8)  # (batch, channels, height, width)
        dtype = torch.float32
        device = 'cpu'
        
        # Create test input tensor
        x = create_test_tensor(input_shape, dtype=dtype, device=device)
        
        # Create a mock return value for fractional_max_pool2d
        # Since fractional max pooling has random behavior, we mock it
        # to return deterministic output for testing
        mock_output = torch.ones(input_shape[0], input_shape[1], output_size[0], output_size[1], 
                                dtype=dtype, device=device) * 0.5
        mock_indices = torch.zeros(input_shape[0], input_shape[1], output_size[0], output_size[1],
                                  dtype=torch.long, device=device)
        
        if return_indices:
            mock_fractional_pool.return_value = (mock_output, mock_indices)
        else:
            mock_fractional_pool.return_value = mock_output
        
        # 1. Test instance creation
        pool_layer = nn.FractionalMaxPool2d(
            kernel_size=kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices
        )
        
        # Assert: instance_created
        assert pool_layer is not None
        assert isinstance(pool_layer, nn.FractionalMaxPool2d)
        
        # Check layer parameters
        assert pool_layer.kernel_size == (kernel_size, kernel_size)  # _pair converts to tuple
        assert pool_layer.output_size == output_size
        assert pool_layer.output_ratio is None  # We specified output_size, not output_ratio
        assert pool_layer.return_indices == return_indices
        
        # 2. Test forward pass
        output = pool_layer(x)
        
        # Assert: forward_works
        assert output is not None
        if return_indices:
            assert isinstance(output, tuple)
            assert len(output) == 2
            output_tensor, indices = output
            assert isinstance(output_tensor, torch.Tensor)
            assert isinstance(indices, torch.Tensor)
        else:
            assert isinstance(output, torch.Tensor)
        
        # 3. Assert: output_shape_correct
        # Fractional max pooling should produce exactly the specified output size
        expected_shape = (input_shape[0], input_shape[1], output_size[0], output_size[1])
        
        if return_indices:
            assert output_tensor.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {output_tensor.shape}"
            assert indices.shape == expected_shape, \
                f"Expected indices shape {expected_shape}, got {indices.shape}"
        else:
            assert output.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {output.shape}"
        
        # 4. Assert: output_dtype_preserved
        if return_indices:
            assert output_tensor.dtype == dtype, \
                f"Expected dtype {dtype}, got {output_tensor.dtype}"
            assert indices.dtype == torch.long, \
                f"Expected indices dtype torch.long, got {indices.dtype}"
        else:
            assert output.dtype == dtype, \
                f"Expected dtype {dtype}, got {output.dtype}"
        
        # 5. Verify that fractional_max_pool2d was called with correct arguments
        mock_fractional_pool.assert_called_once()
        call_args = mock_fractional_pool.call_args
        
        # Check input tensor
        assert torch.equal(call_args[0][0], x)
        
        # Check kernel_size
        assert call_args[0][1] == (kernel_size, kernel_size)
        
        # Check output_size
        assert call_args[0][2] == output_size
        
        # Check output_ratio
        assert call_args[0][3] is None
        
        # Check return_indices
        assert call_args[0][4] == return_indices
        
        # 6. Additional weak assertion: mock was used
        assert mock_fractional_pool.called, \
            "fractional_max_pool2d should have been called"
    
    # ==== BLOCK:CASE_09 START ====
    # Placeholder for LPPool1d basic functionality (DEFERRED)
    # TC-09: LPPool1d基本功能
    # This test case is deferred to later rounds
    # ==== BLOCK:CASE_09 END ====
    
    # ==== BLOCK:CASE_10 START ====
    # Placeholder for MaxUnpool1d basic functionality (DEFERRED)
    # TC-10: MaxUnpool1d基本功能
    # This test case is deferred to later rounds
    # ==== BLOCK:CASE_10 END ====

# ==== BLOCK:FOOTER START ====
    @patch('torch.nn.functional.fractional_max_pool2d')
    def test_fractional_maxpool2d_with_output_ratio(self, mock_fractional_pool, set_random_seed, create_test_tensor):
        """Extension for CASE_08: Test FractionalMaxPool2d with output_ratio parameter
        
        This test extends the basic functionality test to cover the param_extension
        from test_plan.json: output_ratio=[0.5, 0.5], return_indices=True, dtype=float64
        """
        # Test parameters from param_extensions in test_plan.json
        kernel_size = 3
        output_size = None
        output_ratio = (0.5, 0.5)
        return_indices = True
        input_shape = (2, 3, 12, 12)  # (batch, channels, height, width)
        dtype = torch.float64
        device = 'cpu'
        
        # Create test input tensor
        x = create_test_tensor(input_shape, dtype=dtype, device=device)
        
        # Calculate expected output size based on output_ratio
        expected_H_out = int(input_shape[2] * output_ratio[0])
        expected_W_out = int(input_shape[3] * output_ratio[1])
        expected_shape = (input_shape[0], input_shape[1], expected_H_out, expected_W_out)
        
        # Create mock return values
        mock_output = torch.ones(expected_shape, dtype=dtype, device=device) * 0.5
        mock_indices = torch.zeros(expected_shape, dtype=torch.long, device=device)
        mock_fractional_pool.return_value = (mock_output, mock_indices)
        
        # 1. Test instance creation with output_ratio
        pool_layer = nn.FractionalMaxPool2d(
            kernel_size=kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices
        )
        
        # Assert: instance_created
        assert pool_layer is not None
        assert isinstance(pool_layer, nn.FractionalMaxPool2d)
        
        # Check layer parameters
        assert pool_layer.kernel_size == (kernel_size, kernel_size)
        assert pool_layer.output_size is None  # We specified output_ratio, not output_size
        assert pool_layer.output_ratio == output_ratio
        assert pool_layer.return_indices == return_indices
        
        # 2. Test forward pass
        output = pool_layer(x)
        
        # Assert: forward_works
        assert output is not None
        assert isinstance(output, tuple)
        assert len(output) == 2
        output_tensor, indices = output
        assert isinstance(output_tensor, torch.Tensor)
        assert isinstance(indices, torch.Tensor)
        
        # 3. Assert: output_shape_correct
        assert output_tensor.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {output_tensor.shape}"
        assert indices.shape == expected_shape, \
            f"Expected indices shape {expected_shape}, got {indices.shape}"
        
        # 4. Assert: output_dtype_preserved
        assert output_tensor.dtype == dtype, \
            f"Expected dtype {dtype}, got {output_tensor.dtype}"
        assert indices.dtype == torch.long, \
            f"Expected indices dtype torch.long, got {indices.dtype}"
        
        # 5. Verify that fractional_max_pool2d was called with correct arguments
        mock_fractional_pool.assert_called_once()
        call_args = mock_fractional_pool.call_args
        
        # Check input tensor
        assert torch.equal(call_args[0][0], x)
        
        # Check kernel_size
        assert call_args[0][1] == (kernel_size, kernel_size)
        
        # Check output_size (should be None since we use output_ratio)
        assert call_args[0][2] is None
        
        # Check output_ratio
        assert call_args[0][3] == output_ratio
        
        # Check return_indices
        assert call_args[0][4] == return_indices
        
        # 6. Test that mock was used
        assert mock_fractional_pool.called, \
            "fractional_max_pool2d should have been called"
        
        # 7. Additional test: verify output_ratio produces correct size
        # When output_ratio is (0.5, 0.5) and input is 12x12, output should be 6x6
        assert expected_H_out == 6, f"Expected height 6, got {expected_H_out}"
        assert expected_W_out == 6, f"Expected width 6, got {expected_W_out}"
    
    # Additional test for LPPool1d with different norm types
    def test_lppool1d_norm_types(self, set_random_seed):
        """Test LPPool1d with various norm types"""
        test_cases = [
            (1, "sum pooling"),
            (2, "Euclidean norm"),
            (3, "cubic norm"),
            (4, "quartic norm"),
        ]
        
        for norm_type, description in test_cases:
            # Create input with simple values for easy verification
            input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
            
            # Create LPPool1d layer
            pool_layer = nn.LPPool1d(norm_type=norm_type, kernel_size=2, stride=2)
            
            # Apply pooling
            output = pool_layer(input_tensor)
            
            # Verify instance creation
            assert pool_layer is not None
            assert isinstance(pool_layer, nn.LPPool1d)
            assert pool_layer.norm_type == norm_type
            
            # Verify output shape
            assert output.shape == (1, 1, 2), \
                f"Norm type {norm_type} ({description}): expected shape (1, 1, 2), got {output.shape}"
            
            # Verify output values are reasonable
            # Window 1: [1, 2], Window 2: [3, 4]
            if norm_type == 1:
                # Sum pooling: (1+2)=3, (3+4)=7
                expected = torch.tensor([[[3.0, 7.0]]])
            elif norm_type == 2:
                # Euclidean: sqrt(1^2+2^2)=sqrt(5)≈2.236, sqrt(9+16)=sqrt(25)=5
                expected = torch.tensor([[[2.2361, 5.0]]])
            else:
                # For other norm types, just check they're positive and reasonable
                assert torch.all(output > 0), \
                    f"Norm type {norm_type}: output should be positive"
                assert output[0, 0, 0] < 3.5, \
                    f"Norm type {norm_type}: first value should be less than 3.5"
                assert output[0, 0, 1] < 8.0, \
                    f"Norm type {norm_type}: second value should be less than 8.0"
            
            if norm_type <= 2:  # We can compute exact values for norm 1 and 2
                assert torch.allclose(output, expected, rtol=1e-4), \
                    f"Norm type {norm_type} calculation mismatch: got {output}, expected {expected}"
    
    # Additional test for MaxUnpool1d edge cases
    def test_maxunpool1d_edge_cases(self, set_random_seed):
        """Test MaxUnpool1d with various edge cases"""
        # Test 1: Single element input
        single_input = torch.tensor([[[5.0]]], dtype=torch.float32)
        single_pool = nn.MaxPool1d(kernel_size=1, stride=1, return_indices=True)
        single_pooled, single_indices = single_pool(single_input)
        
        single_unpool = nn.MaxUnpool1d(kernel_size=1, stride=1)
        single_unpooled = single_unpool(single_pooled, single_indices)
        
        assert torch.allclose(single_unpooled, single_input), \
            f"Single element unpooling failed: got {single_unpooled}, expected {single_input}"
        
        # Test 2: Input with all same values
        same_input = torch.ones((1, 1, 4), dtype=torch.float32) * 3.0
        same_pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        same_pooled, same_indices = same_pool(same_input)
        
        same_unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
        same_unpooled = same_unpool(same_pooled, same_indices)
        
        # When all values are the same, the first one in each window is chosen
        # Input: [3, 3, 3, 3]
        # MaxPool with kernel=2, stride=2: windows [3,3] and [3,3]
        # Max positions: indices 0 and 2 (first element in each window)
        # Unpooled: [3, 0, 3, 0]
        expected_same = torch.tensor([[[3.0, 0.0, 3.0, 0.0]]], dtype=torch.float32)
        
        assert torch.allclose(same_unpooled, expected_same, rtol=1e-6), \
            f"Same values unpooling failed: got {same_unpooled}, expected {expected_same}"
        
        # Test 3: With output_size parameter different from default
        test_input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=torch.float32)
        test_pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        test_pooled, test_indices = test_pool(test_input)
        
        test_unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
        
        # Test with explicit output_size
        custom_size = (1, 1, 6)  # Larger than default
        custom_unpooled = test_unpool(test_pooled, test_indices, output_size=custom_size)
        
        assert custom_unpooled.shape == custom_size, \
            f"Custom output size failed: got shape {custom_unpooled.shape}, expected {custom_size}"
        
        # The extra positions should be zeros
        assert custom_unpooled[0, 0, -1] == 0.0, \
            "Last position with custom size should be zero"

# ==== BLOCK:FOOTER END ====
# ==== BLOCK:FOOTER END ====