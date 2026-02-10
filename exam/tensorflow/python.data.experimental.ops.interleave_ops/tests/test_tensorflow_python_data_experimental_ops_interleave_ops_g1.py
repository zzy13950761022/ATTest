"""
Test cases for tensorflow.python.data.experimental.ops.interleave_ops - G1 group
Testing parallel_interleave function family
"""
import warnings
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.data.experimental.ops import interleave_ops


# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G1 group
@pytest.fixture
def tf_record_simulated_dataset():
    """Simulate TFRecord dataset for testing."""
    def _create_dataset(size=10):
        data = tf.data.Dataset.range(size)
        return data.map(lambda x: tf.io.serialize_tensor(tf.cast(x, tf.float32)))
    return _create_dataset


@pytest.fixture
def simple_range_dataset():
    """Create simple range dataset for testing."""
    def _create_dataset(size=10):
        return tf.data.Dataset.range(size)
    return _create_dataset


def count_dataset_elements(dataset):
    """Count elements in a dataset."""
    count = 0
    for _ in dataset:
        count += 1
    return count


def capture_deprecation_warnings():
    """Context manager to capture deprecation warnings."""
    return warnings.catch_warnings(record=True)
# ==== BLOCK:HEADER END ====


# ==== BLOCK:CASE_01 START ====
# TC-01: parallel_interleave 基本功能
@pytest.mark.parametrize(
    "cycle_length,block_length,sloppy,map_func_type,dataset_size",
    [
        (2, 1, False, "simple_range", 10),
    ]
)
def test_parallel_interleave_basic(
    cycle_length, block_length, sloppy, map_func_type, dataset_size,
    simple_range_dataset, tf_record_simulated_dataset
):
    """Test basic functionality of parallel_interleave."""
    # Create base dataset
    base_dataset = simple_range_dataset(dataset_size)
    
    # Define map function based on type
    if map_func_type == "simple_range":
        def map_func(x):
            return tf.data.Dataset.range(x + 5)
    elif map_func_type == "tfrecord_simulated":
        def map_func(x):
            return tf_record_simulated_dataset(x + 5)
    else:
        def map_func(x):
            return tf.data.Dataset.from_tensors(x)
    
    # Apply parallel_interleave - deprecation warnings are logged, not captured by warnings module
    # We'll verify the function works correctly instead of checking for warnings
    transform_fn = interleave_ops.parallel_interleave(
        map_func=map_func,
        cycle_length=cycle_length,
        block_length=block_length,
        sloppy=sloppy
    )
    
    # Apply transformation to create dataset
    result_dataset = base_dataset.apply(transform_fn)
    
    # Verify dataset structure
    assert isinstance(result_dataset, tf.data.Dataset)
    
    # Count elements (weak assertion)
    element_count = count_dataset_elements(result_dataset)
    expected_min = dataset_size * 5  # Each input produces at least 5 elements
    assert element_count >= expected_min, f"Expected at least {expected_min} elements, got {element_count}"
    
    # Verify transformation returns callable
    assert callable(transform_fn), "parallel_interleave should return a callable transformation function"
# ==== BLOCK:CASE_01 END ====


# ==== BLOCK:CASE_02 START ====
# TC-02: parallel_interleave 参数边界
@pytest.mark.parametrize(
    "cycle_length,block_length,sloppy,map_func_type,dataset_size",
    [
        (1, 1, True, "identity", 5),
    ]
)
def test_parallel_interleave_boundary_params(
    cycle_length, block_length, sloppy, map_func_type, dataset_size,
    simple_range_dataset
):
    """Test boundary parameters of parallel_interleave."""
    # Create base dataset
    base_dataset = simple_range_dataset(dataset_size)
    
    # Define map function based on type
    if map_func_type == "identity":
        def map_func(x):
            return tf.data.Dataset.from_tensors(x)
    else:
        def map_func(x):
            return tf.data.Dataset.range(x + 1)
    
    # Apply parallel_interleave - deprecation warnings are logged, not captured by warnings module
    # We'll verify the function works correctly instead of checking for warnings
    transform_fn = interleave_ops.parallel_interleave(
        map_func=map_func,
        cycle_length=cycle_length,
        block_length=block_length,
        sloppy=sloppy
    )
    
    # Apply transformation to create dataset
    result_dataset = base_dataset.apply(transform_fn)
    
    # Verify dataset structure
    assert isinstance(result_dataset, tf.data.Dataset)
    
    # Verify no crash with minimal parameters
    # Count elements (weak assertion)
    element_count = count_dataset_elements(result_dataset)
    assert element_count > 0, "Dataset should have elements"
    
    # Verify transformation returns callable
    assert callable(transform_fn), "parallel_interleave should return a callable transformation function"
    
    # Test with cycle_length=1 (minimum valid value)
    # This should work without errors
    try:
        # Consume some elements to ensure no runtime errors
        iterator = iter(result_dataset)
        for _ in range(min(3, element_count)):
            next(iterator)
    except Exception as e:
        pytest.fail(f"Dataset consumption failed with cycle_length=1: {e}")
# ==== BLOCK:CASE_02 END ====


# ==== BLOCK:CASE_05 START ====
# TC-05: parallel_interleave 异常处理
@pytest.mark.parametrize(
    "cycle_length,block_length,map_func_type,expect_error",
    [
        (0, 1, "simple_range", True),
    ]
)
def test_parallel_interleave_error_handling(
    cycle_length, block_length, map_func_type, expect_error,
    simple_range_dataset
):
    """Test error handling for invalid parameters in parallel_interleave."""
    # Create base dataset
    base_dataset = simple_range_dataset(5)
    
    # Define map function
    def map_func(x):
        return tf.data.Dataset.range(x + 5)
    
    # Apply parallel_interleave - deprecation warnings are logged, not captured by warnings module
    # We'll focus on error handling instead of warning capture
    if expect_error:
        # Should raise InvalidArgumentError for cycle_length <= 0
        with pytest.raises(tf.errors.InvalidArgumentError) as exc_info:
            transform_fn = interleave_ops.parallel_interleave(
                map_func=map_func,
                cycle_length=cycle_length,
                block_length=block_length
            )
            # Try to apply if no error raised (should not happen)
            base_dataset.apply(transform_fn)
        
        # Verify error message contains relevant information
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["cycle", "length", "positive", "0", "> 0"])
    else:
        # Should work without error
        transform_fn = interleave_ops.parallel_interleave(
            map_func=map_func,
            cycle_length=cycle_length,
            block_length=block_length
        )
        
        # Apply transformation
        result_dataset = base_dataset.apply(transform_fn)
        assert isinstance(result_dataset, tf.data.Dataset)
# ==== BLOCK:CASE_05 END ====


# ==== BLOCK:CASE_06 START ====
# TC-06: DEFERRED - parallel_interleave 扩展参数
# Placeholder for deferred test case
# ==== BLOCK:CASE_06 END ====


# ==== BLOCK:FOOTER START ====
# Additional helper functions and cleanup for G1 group
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====