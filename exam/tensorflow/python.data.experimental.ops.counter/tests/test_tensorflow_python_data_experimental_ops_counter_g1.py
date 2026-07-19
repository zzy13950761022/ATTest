# ==== BLOCK:HEADER START ====
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental.ops import counter

# Test class for CounterV2 functionality - G1 Group: 核心计数功能
class TestCounterV2G1:
    """Test cases for tensorflow.python.data.experimental.ops.counter.CounterV2 - Core Counting Functionality"""
    
    # Helper method to extract first N elements from a dataset
    def _get_first_n_elements(self, dataset, n=5):
        """Extract first n elements from a dataset as a list."""
        return list(dataset.take(n).as_numpy_iterator())
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# TC-01: 默认参数计数
def test_counter_default_parameters():
    """Test CounterV2 with default parameters (start=0, step=1, dtype=int64)."""
    # Create dataset with default parameters
    dataset = counter.CounterV2()
    
    # Weak assertions
    # 1. dataset_type: Verify it's a Dataset object
    assert isinstance(dataset, tf.data.Dataset)
    
    # 2. element_shape: Verify element spec shape is scalar
    assert dataset.element_spec.shape == ()
    
    # 3. dtype_match: Verify dtype is int64
    assert dataset.element_spec.dtype == tf.int64
    
    # 4. sequence_start: Verify first few elements match expected sequence
    first_five = list(dataset.take(5).as_numpy_iterator())
    expected = [0, 1, 2, 3, 4]
    assert first_five == expected
    
    # Additional verification: dataset can produce more elements
    first_ten = list(dataset.take(10).as_numpy_iterator())
    assert len(first_ten) == 10
    assert first_ten[:5] == expected
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# TC-02: 指定start和step参数
def test_counter_with_start_and_step():
    """Test CounterV2 with specified start=5 and step=2."""
    # Create dataset with specified parameters
    dataset = counter.CounterV2(start=5, step=2, dtype=tf.int64)
    
    # Weak assertions
    # 1. dataset_type: Verify it's a Dataset object
    assert isinstance(dataset, tf.data.Dataset)
    
    # 2. element_shape: Verify element spec shape is scalar
    assert dataset.element_spec.shape == ()
    
    # 3. dtype_match: Verify dtype is int64
    assert dataset.element_spec.dtype == tf.int64
    
    # 4. sequence_correctness: Verify sequence matches arithmetic progression
    first_five = list(dataset.take(5).as_numpy_iterator())
    expected = [5, 7, 9, 11, 13]
    assert first_five == expected
    
    # Verify the arithmetic progression formula: start + n * step
    for i, value in enumerate(first_five):
        assert value == 5 + i * 2
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: 负步长递减计数
def test_counter_negative_step():
    """Test CounterV2 with negative step (decreasing sequence)."""
    # Create dataset with negative step
    dataset = counter.CounterV2(start=10, step=-1, dtype=tf.int64)
    
    # Weak assertions
    # 1. dataset_type: Verify it's a Dataset object
    assert isinstance(dataset, tf.data.Dataset)
    
    # 2. element_shape: Verify element spec shape is scalar
    assert dataset.element_spec.shape == ()
    
    # 3. dtype_match: Verify dtype is int64
    assert dataset.element_spec.dtype == tf.int64
    
    # 4. decreasing_sequence: Verify sequence is decreasing
    first_five = list(dataset.take(5).as_numpy_iterator())
    expected = [10, 9, 8, 7, 6]
    assert first_five == expected
    
    # Verify decreasing property
    for i in range(len(first_five) - 1):
        assert first_five[i] > first_five[i + 1]
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: 浮点参数验证
def test_counter_float_parameters():
    """Test CounterV2 with float parameters (start=0.5, step=0.1, dtype=float32)."""
    # Create dataset with float parameters
    dataset = counter.CounterV2(start=0.5, step=0.1, dtype=tf.float32)
    
    # Weak assertions
    # 1. dataset_type: Verify it's a Dataset object
    assert isinstance(dataset, tf.data.Dataset)
    
    # 2. element_shape: Verify element spec shape is scalar
    assert dataset.element_spec.shape == ()
    
    # 3. dtype_match: Verify dtype is float32
    assert dataset.element_spec.dtype == tf.float32
    
    # 4. float_sequence: Verify first few elements with tolerance
    first_five = list(dataset.take(5).as_numpy_iterator())
    
    # Expected values: 0.5, 0.6, 0.7, 0.8, 0.9
    expected = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Compare with tolerance for floating point values
    for actual, expected_val in zip(first_five, expected):
        # Use appropriate tolerance for float32
        assert abs(actual - expected_val) < 1e-5
    
    # Verify the arithmetic progression formula
    for i, value in enumerate(first_five):
        expected_val = 0.5 + i * 0.1
        assert abs(value - expected_val) < 1e-5
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:FOOTER START ====
# Additional helper functions and fixtures for G1 group

# Fixture for common test setup
@pytest.fixture
def default_counter():
    """Fixture providing default CounterV2 dataset."""
    return counter.CounterV2()

# Test for independent instances
def test_counter_independent_instances():
    """Test that multiple CounterV2 calls create independent datasets."""
    dataset1 = counter.CounterV2(start=0, step=1)
    dataset2 = counter.CounterV2(start=0, step=1)
    
    # Take elements from both datasets
    elements1 = list(dataset1.take(3).as_numpy_iterator())
    elements2 = list(dataset2.take(3).as_numpy_iterator())
    
    # Both should produce the same sequence
    assert elements1 == [0, 1, 2]
    assert elements2 == [0, 1, 2]
    
    # Create new iterators for both datasets
    # Note: Each call to take() creates a new iterator starting from the beginning
    more_elements1 = list(dataset1.take(5).as_numpy_iterator())
    elements2_again = list(dataset2.take(3).as_numpy_iterator())
    
    # Both should start from beginning when creating new iterators
    assert more_elements1[:3] == [0, 1, 2]  # First 3 elements should match
    assert len(more_elements1) == 5  # Should have 5 elements total
    assert elements2_again == [0, 1, 2]  # Should still start from beginning

# Test for param extension: large start and step values
def test_counter_large_values():
    """Test CounterV2 with large start and step values (param extension for CASE_01)."""
    dataset = counter.CounterV2(start=100, step=10, dtype=tf.int64)
    
    # Verify dataset properties
    assert isinstance(dataset, tf.data.Dataset)
    assert dataset.element_spec.shape == ()
    assert dataset.element_spec.dtype == tf.int64
    
    # Verify sequence
    first_five = list(dataset.take(5).as_numpy_iterator())
    expected = [100, 110, 120, 130, 140]
    assert first_five == expected

# Test for param extension: negative start value
def test_counter_negative_start():
    """Test CounterV2 with negative start value (param extension for CASE_02)."""
    dataset = counter.CounterV2(start=-5, step=3, dtype=tf.int64)
    
    # Verify dataset properties
    assert isinstance(dataset, tf.data.Dataset)
    assert dataset.element_spec.shape == ()
    assert dataset.element_spec.dtype == tf.int64
    
    # Verify sequence
    first_five = list(dataset.take(5).as_numpy_iterator())
    expected = [-5, -2, 1, 4, 7]
    assert first_five == expected

# Test for param extension: large negative step
def test_counter_large_negative_step():
    """Test CounterV2 with large negative step (param extension for CASE_04)."""
    dataset = counter.CounterV2(start=100, step=-10, dtype=tf.int64)
    
    # Verify dataset properties
    assert isinstance(dataset, tf.data.Dataset)
    assert dataset.element_spec.shape == ()
    assert dataset.element_spec.dtype == tf.int64
    
    # Verify sequence
    first_five = list(dataset.take(5).as_numpy_iterator())
    expected = [100, 90, 80, 70, 60]
    assert first_five == expected
# ==== BLOCK:FOOTER END ====