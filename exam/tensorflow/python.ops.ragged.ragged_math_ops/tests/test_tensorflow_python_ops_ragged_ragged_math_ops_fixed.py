"""
Test cases for tensorflow.python.ops.ragged.ragged_math_ops
"""
import math
import numpy as np
import pytest
import tensorflow as tf
from unittest import mock

# Import the target module
from tensorflow.python.ops.ragged import ragged_math_ops

# ==============================================================================
# Helper functions and fixtures
# ==============================================================================

def create_ragged_tensor(values, row_splits=None, dtype=None):
    """Helper to create RaggedTensor for testing."""
    if row_splits is None:
        # Create from nested list
        return tf.ragged.constant(values, dtype=dtype)
    else:
        # Create from values and row_splits
        return tf.RaggedTensor.from_row_splits(values, row_splits, validate=False)

def assert_ragged_tensor_equal(rt1, rt2, rtol=1e-6, atol=1e-6):
    """Assert two RaggedTensors are equal within tolerance."""
    # Check shape
    assert rt1.shape == rt2.shape
    
    # Check dtype
    assert rt1.dtype == rt2.dtype
    
    # Check values
    if rt1.dtype.is_floating:
        np.testing.assert_allclose(
            rt1.flat_values.numpy(), 
            rt2.flat_values.numpy(),
            rtol=rtol, 
            atol=atol
        )
    else:
        np.testing.assert_array_equal(
            rt1.flat_values.numpy(), 
            rt2.flat_values.numpy()
        )
    
    # Check row splits
    np.testing.assert_array_equal(
        rt1.row_splits.numpy(), 
        rt2.row_splits.numpy()
    )

# ==============================================================================
# Test class
# ==============================================================================

class TestRaggedMathOps:
    """Test class for ragged_math_ops module."""
    
    # ===== BLOCK:HEADER START =====
    def setup_method(self):
        """Setup method for each test."""
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Common test data
        self.test_int32 = tf.int32
        self.test_int64 = tf.int64
        self.test_float32 = tf.float32
        self.test_float64 = tf.float64
    # ===== BLOCK:HEADER END =====
    
    # ===== BLOCK:CASE_01 START =====
    @pytest.mark.parametrize(
        "starts,limits,deltas,dtype,row_splits_dtype",
        [
            (0, 5, 1, "int32", "int32"),
            (2, 10, 2, "int64", "int64"),
        ]
    )
    def test_range_basic_functionality(self, starts, limits, deltas, dtype, row_splits_dtype):
        """Test basic functionality of ragged.range function (TC-01)."""
        # Convert string dtypes to tf dtypes
        tf_dtype = getattr(tf, dtype)
        tf_row_splits_dtype = getattr(tf, row_splits_dtype)
        
        # Create expected Python range
        expected_values = list(range(starts, limits, deltas))
        
        # Call ragged.range with correct parameter names
        result = ragged_math_ops.range(
            starts=starts,
            limits=limits,
            deltas=deltas,
            dtype=tf_dtype,
            row_splits_dtype=tf_row_splits_dtype
        )
        
        # Weak assertions (shape, dtype, values_match, row_splits_correct)
        # 1. Check shape - ragged.range returns shape (1, None) for scalar inputs
        # We need to check the rank and first dimension
        assert result.shape.rank == 2
        assert result.shape[0] == 1  # First dimension should be 1 for scalar inputs
        
        # 2. Check dtype
        assert result.dtype == tf_dtype
        
        # 3. Check values match Python range
        actual_values = result.values.numpy().tolist()
        assert actual_values == expected_values
        
        # 4. Check row_splits are correct
        expected_row_splits = [0, len(expected_values)]
        actual_row_splits = result.row_splits.numpy().tolist()
        assert actual_row_splits == expected_row_splits
        
        # 5. Check row_splits dtype
        assert result.row_splits.dtype == tf_row_splits_dtype
    # ===== BLOCK:CASE_01 END =====
    
    # ===== BLOCK:CASE_02 START =====
    @pytest.mark.parametrize(
        "input_shape,axis,dtype",
        [
            ([[2, 3], [4, 5]], 1, "float32"),
            ([[1, 2, 3], [4, 5]], 0, "float64"),
        ]
    )
    def test_reduce_sum_single_axis(self, input_shape, axis, dtype):
        """Test single-axis reduction with reduce_sum (TC-02)."""
        # Create test RaggedTensor
        tf_dtype = getattr(tf, dtype)
        rt = tf.ragged.constant(input_shape, dtype=tf_dtype)
        
        # Call reduce_sum
        result = ragged_math_ops.reduce_sum(
            input_tensor=rt,
            axis=axis,
            name='test_reduce_sum'
        )
        
        # Weak assertions (shape, dtype, sum_correctness, finite)
        # 1. Check shape - reduced along specified axis
        if axis == 0:
            # Reduce along first axis (ragged dimension)
            # Result should be a regular tensor with shape (rt.shape[1],)
            assert result.shape.rank == 1
            # For ragged tensor with shape (2, None), reducing axis=0 gives shape (None,)
            # We can't assert exact shape since it's ragged
            assert isinstance(result, tf.Tensor)  # Should be regular tensor after reduction
        else:
            # Reduce along second axis (inner dimension)
            # Result should be a ragged tensor with shape (rt.shape[0],)
            assert result.shape.rank == 1
            assert result.shape[0] == rt.shape[0]
            assert isinstance(result, tf.RaggedTensor)  # Should still be ragged
        
        # 2. Check dtype is preserved
        assert result.dtype == tf_dtype
        
        # 3. Check sum correctness (basic check)
        # For ragged tensors, sum should not produce NaN/Inf
        if isinstance(result, tf.RaggedTensor):
            values = result.flat_values.numpy()
        else:
            values = result.numpy()
        assert np.all(np.isfinite(values))
        
        # 4. Check that values are reasonable (not all zeros unless input is zero)
        # Sum of positive numbers should be positive
        if tf_dtype.is_floating:
            assert np.any(values != 0.0)
        
        # Additional weak assertion: result is a valid tensor
        assert isinstance(result, (tf.Tensor, tf.RaggedTensor))
    # ===== BLOCK:CASE_02 END =====
    
    # ===== BLOCK:CASE_03 START =====
    @pytest.mark.parametrize(
        "data_shape,segment_ids,dtype",
        [
            ([5, 3], [0, 0, 1, 1, 2], "float32"),
            ([3, 2], [0, 1, 0], "int32"),
        ]
    )
    def test_segment_sum_basic_aggregation(self, data_shape, segment_ids, dtype):
        """Test basic aggregation with segment_sum (TC-03)."""
        # Create test data
        tf_dtype = getattr(tf, dtype)
        
        # Create a regular tensor for testing (segment_sum works with both regular and ragged tensors)
        # Use deterministic values for reproducibility
        np.random.seed(42)
        if tf_dtype.is_floating:
            data_values = np.random.randn(*data_shape).astype(np.float32 if dtype == "float32" else np.float64)
        else:
            # For integer types, use small integer values
            data_values = np.random.randint(0, 10, size=data_shape).astype(np.int32)
        
        data = tf.constant(data_values, dtype=tf_dtype)
        
        # Create segment_ids tensor
        segment_ids_tensor = tf.constant(segment_ids, dtype=tf.int32)
        
        # Calculate num_segments
        num_segments = max(segment_ids) + 1
        
        # Call segment_sum
        result = ragged_math_ops.segment_sum(
            data=data,
            segment_ids=segment_ids_tensor,
            num_segments=num_segments,
            name='test_segment_sum'
        )
        
        # Weak assertions (shape, dtype, segment_aggregation, segment_ids_valid)
        # 1. Check shape - should have num_segments rows
        assert result.shape[0] == num_segments
        
        # 2. Check dtype is preserved
        assert result.dtype == tf_dtype
        
        # 3. Check segment aggregation correctness
        # Manually compute expected segment sum
        expected_result = np.zeros((num_segments, data_shape[1]), 
                                  dtype=data_values.dtype)
        for i, seg_id in enumerate(segment_ids):
            expected_result[seg_id] += data_values[i]
        
        # Compare with actual result
        np.testing.assert_array_almost_equal(
            result.numpy(), 
            expected_result,
            decimal=5 if tf_dtype.is_floating else 0
        )
        
        # 4. Check segment_ids are valid (non-negative)
        assert all(sid >= 0 for sid in segment_ids)
        assert max(segment_ids) < num_segments
        
        # Additional weak assertion: result is a tensor
        assert isinstance(result, tf.Tensor)
    # ===== BLOCK:CASE_03 END =====
    
    # ===== BLOCK:CASE_04 START =====
    # DEFERRED: matmul混合运算 - will be implemented in later rounds
    @pytest.mark.skip(reason="Deferred to later round")
    def test_matmul_mixed_operations(self):
        """Placeholder for matmul mixed operations test (TC-04)."""
        pass
    # ===== BLOCK:CASE_04 END =====
    
    # ===== BLOCK:CASE_05 START =====
    # DEFERRED: dropout随机性控制 - will be implemented in later rounds
    @pytest.mark.skip(reason="Deferred to later round")
    def test_dropout_randomness_control(self):
        """Placeholder for dropout randomness control test (TC-05)."""
        pass
    # ===== BLOCK:CASE_05 END =====
    
    # ===== BLOCK:FOOTER START =====
    def test_edge_case_empty_ragged_tensor(self):
        """Test edge case: empty RaggedTensor handling."""
        # Create empty RaggedTensor
        empty_rt = tf.ragged.constant([], dtype=tf.float32)
        
        # Test reduce_sum on empty tensor
        # For empty ragged tensor, reduce_sum should handle it gracefully
        # The exact behavior may vary, so we test for graceful handling
        try:
            result = ragged_math_ops.reduce_sum(empty_rt, axis=0)
            # If it succeeds, check it's a valid tensor
            assert isinstance(result, (tf.Tensor, tf.RaggedTensor))
        except (ValueError, tf.errors.InvalidArgumentError) as e:
            # It's also acceptable to raise an error for empty input
            assert "empty" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_invalid_segment_ids(self):
        """Test invalid segment_ids raise appropriate error."""
        data = tf.ragged.constant([[1, 2], [3, 4, 5]], dtype=tf.float32)
        invalid_segment_ids = tf.constant([0, 0, 2], dtype=tf.int32)  # Wrong length
        
        # This should raise ValueError because segment_ids.shape must be prefix of data.shape
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            ragged_math_ops.segment_sum(
                data=data,
                segment_ids=invalid_segment_ids,
                num_segments=3
            )
    
    def test_range_with_negative_delta(self):
        """Test range function with negative delta."""
        # Use correct parameter names: starts, limits, deltas
        result = ragged_math_ops.range(
            starts=5,
            limits=0,
            deltas=-1,
            dtype=tf.int32
        )
        
        expected = list(range(5, 0, -1))
        assert result.values.numpy().tolist() == expected
    # ===== BLOCK:FOOTER END =====