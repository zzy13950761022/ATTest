"""
Test cases for tensorflow.python.data.experimental.ops.prefetching_ops
Group G1: prefetch_to_device核心功能
"""
import math
import pytest
import tensorflow as tf
from tensorflow.python.data.experimental.ops.prefetching_ops import prefetch_to_device, copy_to_device

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions
@pytest.fixture
def tf_seed():
    """Set TensorFlow random seed for reproducibility."""
    tf.random.set_seed(42)
    return 42

def create_tensor_slices_dataset(data_shape, dtype, num_elements=None):
    """Create a tf.data.Dataset from tensor slices."""
    if num_elements is None:
        num_elements = data_shape[0] if data_shape else 1
    
    if dtype == "int32":
        data = tf.range(num_elements, dtype=tf.int32)
    elif dtype == "float32":
        data = tf.range(num_elements, dtype=tf.float32)
    elif dtype == "float64":
        data = tf.range(num_elements, dtype=tf.float64)
    elif dtype == "int64":
        data = tf.range(num_elements, dtype=tf.int64)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Reshape if needed
    if len(data_shape) > 1:
        total_elements = 1
        for dim in data_shape:
            total_elements *= dim
        data = tf.reshape(tf.range(total_elements, dtype=getattr(tf, dtype)), data_shape)
    
    return tf.data.Dataset.from_tensor_slices(data)

def create_empty_dataset(dtype="int32"):
    """Create an empty dataset."""
    if dtype == "int32":
        data = tf.constant([], dtype=tf.int32)
    else:
        data = tf.constant([], dtype=getattr(tf, dtype))
    return tf.data.Dataset.from_tensor_slices(data)
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# TC-01: prefetch_to_device基本功能验证
# Priority: High, Group: G1
# Parameters: device=/cpu:0, buffer_size=1, dataset_type=tensor_slices, data_shape=[10], dtype=int32
# Weak asserts: returns_callable, dataset_apply_compatible, output_shape_match, output_dtype_match
@pytest.mark.parametrize("device,buffer_size,data_shape,dtype", [
    ("/cpu:0", 1, [10], "int32"),
])
def test_prefetch_to_device_basic_functionality(device, buffer_size, data_shape, dtype):
    """Test basic functionality of prefetch_to_device."""
    # Create dataset
    dataset = create_tensor_slices_dataset(data_shape, dtype)
    
    # Get transformation function
    transform_fn = prefetch_to_device(device=device, buffer_size=buffer_size)
    
    # Assert 1: returns_callable - transformation function should be callable
    assert callable(transform_fn), "prefetch_to_device should return a callable function"
    
    # Apply transformation
    transformed_dataset = dataset.apply(transform_fn)
    
    # Assert 2: dataset_apply_compatible - should work with Dataset.apply
    assert isinstance(transformed_dataset, tf.data.Dataset), \
        "Transformed result should be a tf.data.Dataset"
    
    # Create iterator and get first element
    iterator = iter(transformed_dataset)
    first_element = next(iterator)
    
    # Assert 3: output_shape_match - shape should match input
    expected_shape = data_shape[1:] if len(data_shape) > 1 else ()
    assert first_element.shape == expected_shape, \
        f"Output shape {first_element.shape} should match expected {expected_shape}"
    
    # Assert 4: output_dtype_match - dtype should match input
    expected_dtype = getattr(tf, dtype)
    assert first_element.dtype == expected_dtype, \
        f"Output dtype {first_element.dtype} should match expected {expected_dtype}"
    
    # Verify we can iterate through all elements
    count = 1
    for _ in iterator:
        count += 1
    
    expected_count = data_shape[0] if data_shape else 1
    assert count == expected_count, \
        f"Should iterate through {expected_count} elements, got {count}"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# TC-02: prefetch_to_device buffer_size参数验证
# Priority: High, Group: G1
# Parameters: device=/cpu:0, buffer_size=None, dataset_type=tensor_slices, data_shape=[5], dtype=float32
# Weak asserts: returns_callable, dataset_apply_compatible, auto_buffer_size_works, no_crash
@pytest.mark.parametrize("device,buffer_size,data_shape,dtype", [
    ("/cpu:0", None, [5], "float32"),
])
def test_prefetch_to_device_buffer_size_parameter(device, buffer_size, data_shape, dtype):
    """Test prefetch_to_device with auto buffer_size selection."""
    # Create dataset
    dataset = create_tensor_slices_dataset(data_shape, dtype)
    
    # Get transformation function with auto buffer_size
    transform_fn = prefetch_to_device(device=device, buffer_size=buffer_size)
    
    # Assert 1: returns_callable - transformation function should be callable
    assert callable(transform_fn), "prefetch_to_device should return a callable function"
    
    # Apply transformation
    transformed_dataset = dataset.apply(transform_fn)
    
    # Assert 2: dataset_apply_compatible - should work with Dataset.apply
    assert isinstance(transformed_dataset, tf.data.Dataset), \
        "Transformed result should be a tf.data.Dataset"
    
    # Assert 3: auto_buffer_size_works - should work without explicit buffer_size
    # Create iterator and get elements
    iterator = iter(transformed_dataset)
    
    # Get all elements to verify no crash
    elements = []
    for element in iterator:
        elements.append(element)
    
    # Assert 4: no_crash - should complete without errors
    assert len(elements) == data_shape[0], \
        f"Should get {data_shape[0]} elements, got {len(elements)}"
    
    # Verify dtype matches
    expected_dtype = getattr(tf, dtype)
    assert all(elem.dtype == expected_dtype for elem in elements), \
        "All elements should have correct dtype"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: prefetch_to_device无效参数处理 (DEFERRED - placeholder)
# Priority: Medium, Group: G1
# This test case is deferred and will be implemented in later rounds.
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: prefetch_to_device空数据集处理 (DEFERRED - placeholder)
# Priority: Medium, Group: G1
# This test case is deferred and will be implemented in later rounds.
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: copy_to_device基本功能验证 (DEFERRED - G2 test, placeholder for reference)
# Priority: High, Group: G2
# This test case belongs to group G2 and will be implemented in its own file.
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:FOOTER START ====
# Additional test utilities and cleanup
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====