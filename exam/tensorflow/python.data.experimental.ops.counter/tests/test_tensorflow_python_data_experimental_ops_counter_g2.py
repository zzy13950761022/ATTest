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
    (tf.int64, tf.int64),  # Add default dtype test
    (tf.float32, tf.float32),  # Add float32 test
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
        # Convert numpy types to Python types for comparison
        first_five_py = [int(x) if hasattr(x, 'item') else x for x in first_five]
        assert first_five_py == expected
    elif expected_dtype in [tf.float32, tf.float64]:
        # For float types, we need to compare with tolerance
        expected = [0.0, 1.0, 2.0, 3.0, 4.0]
        for actual, expected_val in zip(first_five, expected):
            # Use appropriate tolerance for float32 vs float64
            tolerance = 1e-6 if expected_dtype == tf.float32 else 1e-10
            assert abs(actual - expected_val) < tolerance
    
    # Verify dtype of elements - adjust for numpy types
    for element in first_five:
        # Check that element has appropriate type
        if expected_dtype in [tf.int32, tf.int64]:
            # TensorFlow int elements come as numpy.int32/int64 via as_numpy_iterator()
            # We can check using numpy type or convert to Python int
            assert hasattr(element, 'item') or isinstance(element, (int, np.integer))
        elif expected_dtype == tf.float32:
            # TensorFlow float32 elements come as numpy.float32 via as_numpy_iterator()
            assert hasattr(element, 'item') or isinstance(element, (float, np.floating))
            # Additional check for float32 precision
            assert np.float32 == type(element) or isinstance(element, np.float32)
        elif expected_dtype == tf.float64:
            # TensorFlow float64 elements come as numpy.float64 via as_numpy_iterator()
            assert hasattr(element, 'item') or isinstance(element, (float, np.floating))
    
    # Strong assertions (enabled in final stage)
    # 1. float_dtype_support: Verify float types work correctly
    if expected_dtype in [tf.float32, tf.float64]:
        # Test with fractional step
        float_dataset = counter.CounterV2(start=0.5, step=0.1, dtype=expected_dtype)
        float_elements = list(float_dataset.take(3).as_numpy_iterator())
        expected_float = [0.5, 0.6, 0.7]
        for actual, expected_val in zip(float_elements, expected_float):
            tolerance = 1e-6 if expected_dtype == tf.float32 else 1e-10
            assert abs(actual - expected_val) < tolerance
    
    # 2. type_conversion: Verify start/step are converted to specified dtype
    # Test with integer start/step but float dtype
    if expected_dtype in [tf.float32, tf.float64]:
        mixed_dataset = counter.CounterV2(start=1, step=2, dtype=expected_dtype)
        mixed_elements = list(mixed_dataset.take(3).as_numpy_iterator())
        expected_mixed = [1.0, 3.0, 5.0]
        for actual, expected_val in zip(mixed_elements, expected_mixed):
            tolerance = 1e-6 if expected_dtype == tf.float32 else 1e-10
            assert abs(actual - expected_val) < tolerance
    
    # 3. precision_limits: Test with extreme values
    if expected_dtype == tf.float32:
        # Test with values near float32 limits
        large_dataset = counter.CounterV2(start=1e6, step=1e6, dtype=tf.float32)
        large_elements = list(large_dataset.take(3).as_numpy_iterator())
        # Verify they are float32
        for element in large_elements:
            assert isinstance(element, np.float32) or np.float32 == type(element)
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
    
    # Strong assertions (enabled in final stage)
    # 1. Test with fractional values
    float_dataset = counter.CounterV2(start=0.5, step=0.25, dtype=tf.float64)
    float_elements = list(float_dataset.take(4).as_numpy_iterator())
    expected_float = [0.5, 0.75, 1.0, 1.25]
    
    for actual, expected_val in zip(float_elements, expected_float):
        assert abs(actual - expected_val) < 1e-10
    
    # 2. Test with large values
    large_dataset = counter.CounterV2(start=1e10, step=1e10, dtype=tf.float64)
    large_elements = list(large_dataset.take(3).as_numpy_iterator())
    expected_large = [1e10, 2e10, 3e10]
    
    for actual, expected_val in zip(large_elements, expected_large):
        # Use relative tolerance for large values
        if expected_val != 0:
            assert abs((actual - expected_val) / expected_val) < 1e-10

# Additional test for int64 default dtype (coverage gap)
def test_counter_default_int64_dtype():
    """Test CounterV2 with default int64 dtype (coverage gap)."""
    # Test without specifying dtype (should default to int64)
    dataset = counter.CounterV2()
    
    # Verify dataset properties
    assert isinstance(dataset, tf.data.Dataset)
    assert dataset.element_spec.shape == ()
    assert dataset.element_spec.dtype == tf.int64
    
    # Verify sequence
    first_five = list(dataset.take(5).as_numpy_iterator())
    expected = [0, 1, 2, 3, 4]
    
    # Convert numpy types to Python int for comparison
    first_five_py = [int(x) if hasattr(x, 'item') else x for x in first_five]
    assert first_five_py == expected
    
    # Strong assertions
    # 1. Test that default dtype is indeed int64
    dataset2 = counter.CounterV2(start=10, step=5)
    assert dataset2.element_spec.dtype == tf.int64
    
    # 2. Test with negative values
    neg_dataset = counter.CounterV2(start=-5, step=-2)
    neg_elements = list(neg_dataset.take(3).as_numpy_iterator())
    expected_neg = [-5, -7, -9]
    neg_elements_py = [int(x) if hasattr(x, 'item') else x for x in neg_elements]
    assert neg_elements_py == expected_neg

# Test for float32 specific behavior
def test_counter_float32_specific():
    """Test CounterV2 with float32 data type specific behaviors."""
    dataset = counter.CounterV2(start=0, step=1, dtype=tf.float32)
    
    # Verify dataset properties
    assert isinstance(dataset, tf.data.Dataset)
    assert dataset.element_spec.shape == ()
    assert dataset.element_spec.dtype == tf.float32
    
    # Verify sequence and element types
    first_five = list(dataset.take(5).as_numpy_iterator())
    expected = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    for actual, expected_val in zip(first_five, expected):
        assert abs(actual - expected_val) < 1e-6
        # Verify it's actually float32
        assert isinstance(actual, np.float32) or np.float32 == type(actual)
    
    # Strong assertions for float32 precision
    # Test with values that might show float32 precision limits
    precision_dataset = counter.CounterV2(start=1e6, step=1, dtype=tf.float32)
    precision_elements = list(precision_dataset.take(3).as_numpy_iterator())
    
    # Verify they maintain float32 type
    for element in precision_elements:
        assert isinstance(element, np.float32) or np.float32 == type(element)
# ==== BLOCK:FOOTER END ====