# ==== BLOCK:HEADER START ====
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental.ops import counter

# Test class for CounterV2 functionality - G2 Group: 数据类型与边界
class TestCounterV2G2:
    """Test cases for tensorflow.python.data.experimental.ops.counter.CounterV2 - Data Types & Boundaries"""
    
    # Helper method to extract first N elements from a dataset
    def _get_first_n_elements(self, dataset, n=5):
        """Extract first n elements from a dataset as a list."""
        return list(dataset.take(n).as_numpy_iterator())
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: 不同数据类型验证
@pytest.mark.parametrize("dtype,expected_dtype", [
    (tf.int32, tf.int32),
    (tf.float64, tf.float64),
])
def test_counter_different_dtypes(dtype, expected_dtype):
    """Test CounterV2 with different data types."""
    # Create dataset with specified dtype
    dataset = counter.CounterV2(start=0, step=1, dtype=dtype)
    
    # Weak assertions
    # 1. dataset_type: Verify it's a Dataset object
    assert isinstance(dataset, tf.data.Dataset)
    
    # 2. element_shape: Verify element spec shape is scalar
    assert dataset.element_spec.shape == ()
    
    # 3. dtype_match: Verify dtype matches expected
    assert dataset.element_spec.dtype == expected_dtype
    
    # 4. sequence_start: Verify first few elements
    first_five = list(dataset.take(5).as_numpy_iterator())
    
    # Expected values depend on dtype
    if expected_dtype in [tf.int32, tf.int64]:
        expected = [0, 1, 2, 3, 4]
        assert first_five == expected
    elif expected_dtype == tf.float64:
        # For float64, we need to compare with tolerance
        expected = [0.0, 1.0, 2.0, 3.0, 4.0]
        for actual, expected_val in zip(first_five, expected):
            assert abs(actual - expected_val) < 1e-10
    
    # Verify dtype of elements
    for element in first_five:
        # Check that element has appropriate type
        if expected_dtype == tf.int32:
            # TensorFlow int32 elements come as Python int
            assert isinstance(element, int)
        elif expected_dtype == tf.float64:
            # TensorFlow float64 elements come as Python float
            assert isinstance(element, float)
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:FOOTER START ====
# Additional helper functions and fixtures for G2 group

# Fixture for common test setup
@pytest.fixture
def default_counter():
    """Fixture providing default CounterV2 dataset."""
    return counter.CounterV2()

# Test for float32 data type (from param_extensions)
def test_counter_float64_dtype():
    """Test CounterV2 with float64 data type (param extension for CASE_03)."""
    dataset = counter.CounterV2(start=0, step=1, dtype=tf.float64)
    
    # Verify dataset properties
    assert isinstance(dataset, tf.data.Dataset)
    assert dataset.element_spec.shape == ()
    assert dataset.element_spec.dtype == tf.float64
    
    # Verify sequence
    first_five = list(dataset.take(5).as_numpy_iterator())
    expected = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    for actual, expected_val in zip(first_five, expected):
        assert abs(actual - expected_val) < 1e-10
    
    # Verify element types
    for element in first_five:
        assert isinstance(element, float)
# ==== BLOCK:FOOTER END ====