"""
Test cases for tensorflow.python.feature_column.sequence_feature_column
Group G2: 上下文拼接与形状处理
"""

import numpy as np
import pytest
import tensorflow as tf

# Import target functions
from tensorflow.python.feature_column.sequence_feature_column import (
    concatenate_context_input
)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==== BLOCK:HEADER START ====
# Test class and fixtures for G2 tests
class TestSequenceFeatureColumnG2:
    """Test cases for context concatenation and shape handling."""
    
    @pytest.fixture
    def random_float32_tensor(self):
        """Create random float32 tensor with given shape."""
        def _create_tensor(shape):
            return tf.constant(np.random.randn(*shape).astype(np.float32))
        return _create_tensor
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_05 START ====
    def test_concatenate_context_input_basic(self, random_float32_tensor):
        """Test basic context input concatenation (CASE_05)."""
        # Test case 1: Basic shape from test plan
        context_shape = (2, 3)  # [batch_size, d1]
        sequence_shape = (2, 5, 4)  # [batch_size, padded_length, d0]
        expected_shape = (2, 5, 7)  # [batch_size, padded_length, d0 + d1]
        
        context_input = random_float32_tensor(context_shape)
        sequence_input = random_float32_tensor(sequence_shape)
        
        # Call the function
        result = concatenate_context_input(context_input, sequence_input)
        
        # Weak assertions
        # 1. Check output shape
        assert result.shape == expected_shape
        
        # 2. Check output dtype
        assert result.dtype == tf.float32
        
        # 3. Check no NaN values
        assert not tf.reduce_any(tf.math.is_nan(result))
        
        # 4. Check no infinite values
        assert not tf.reduce_any(tf.math.is_inf(result))
        
        # Test case 2: Parameter extension from test plan
        context_shape_ext = (4, 2)  # [batch_size, d1]
        sequence_shape_ext = (4, 8, 6)  # [batch_size, padded_length, d0]
        expected_shape_ext = (4, 8, 8)  # [batch_size, padded_length, d0 + d1]
        
        context_input_ext = random_float32_tensor(context_shape_ext)
        sequence_input_ext = random_float32_tensor(sequence_shape_ext)
        
        # Call the function with extended parameters
        result_ext = concatenate_context_input(context_input_ext, sequence_input_ext)
        
        # Check output shape for extended case
        assert result_ext.shape == expected_shape_ext
        
        # Additional basic checks
        # Check that batch dimension is preserved
        assert result.shape[0] == context_shape[0] == sequence_shape[0]
        
        # Check that sequence length is preserved
        assert result.shape[1] == sequence_shape[1]
        
        # Check that concatenation dimension is correct
        assert result.shape[2] == sequence_shape[2] + context_shape[1]
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
    def test_concatenate_context_input_different_shapes(self, random_float32_tensor):
        """Test context input concatenation with different shapes (CASE_06)."""
        # Test case 1: Basic shape from test plan (corrected from [1, 10, 2] to [1, 2])
        context_shape = (1, 2)  # [batch_size, d1] - corrected from test plan
        sequence_shape = (1, 10, 3)  # [batch_size, padded_length, d0]
        expected_shape = (1, 10, 5)  # [batch_size, padded_length, d0 + d1]
        
        context_input = random_float32_tensor(context_shape)
        sequence_input = random_float32_tensor(sequence_shape)
        
        # Call the function
        result = concatenate_context_input(context_input, sequence_input)
        
        # Weak assertions
        # 1. Check output shape
        assert result.shape == expected_shape
        
        # 2. Check batch dimension match
        assert result.shape[0] == context_shape[0] == sequence_shape[0]
        
        # 3. Check sequence dimension match
        assert result.shape[1] == sequence_shape[1]
        
        # Additional test cases for different shapes
        # Test case 2: Larger batch size
        context_shape2 = (4, 3)  # [batch_size, d1]
        sequence_shape2 = (4, 7, 5)  # [batch_size, padded_length, d0]
        expected_shape2 = (4, 7, 8)  # [batch_size, padded_length, d0 + d1]
        
        context_input2 = random_float32_tensor(context_shape2)
        sequence_input2 = random_float32_tensor(sequence_shape2)
        
        result2 = concatenate_context_input(context_input2, sequence_input2)
        assert result2.shape == expected_shape2
        
        # Test case 3: Single feature in context
        context_shape3 = (2, 1)  # [batch_size, d1]
        sequence_shape3 = (2, 3, 4)  # [batch_size, padded_length, d0]
        expected_shape3 = (2, 3, 5)  # [batch_size, padded_length, d0 + d1]
        
        context_input3 = random_float32_tensor(context_shape3)
        sequence_input3 = random_float32_tensor(sequence_shape3)
        
        result3 = concatenate_context_input(context_input3, sequence_input3)
        assert result3.shape == expected_shape3
        
        # Test case 4: Equal dimensions
        context_shape4 = (3, 5)  # [batch_size, d1]
        sequence_shape4 = (3, 8, 5)  # [batch_size, padded_length, d0]
        expected_shape4 = (3, 8, 10)  # [batch_size, padded_length, d0 + d1]
        
        context_input4 = random_float32_tensor(context_shape4)
        sequence_input4 = random_float32_tensor(sequence_shape4)
        
        result4 = concatenate_context_input(context_input4, sequence_input4)
        assert result4.shape == expected_shape4
        
        # Check that all results have correct dtype
        assert result.dtype == tf.float32
        assert result2.dtype == tf.float32
        assert result3.dtype == tf.float32
        assert result4.dtype == tf.float32
        
        # Check no NaN or infinite values
        for res in [result, result2, result3, result4]:
            assert not tf.reduce_any(tf.math.is_nan(res))
            assert not tf.reduce_any(tf.math.is_inf(res))
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:FOOTER START ====
# Additional helper functions and cleanup
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====