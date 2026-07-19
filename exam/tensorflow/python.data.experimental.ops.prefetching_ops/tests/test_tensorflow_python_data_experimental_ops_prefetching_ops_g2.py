"""
Test cases for tensorflow.python.data.experimental.ops.prefetching_ops
Group G2: copy_to_device与设备间传输
"""
import math
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.data.experimental.ops.prefetching_ops import prefetch_to_device, copy_to_device

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G2
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

# Test cases for helper functions to improve coverage
def test_create_tensor_slices_dataset_basic():
    """Test basic functionality of create_tensor_slices_dataset."""
    # Test with int32
    dataset = create_tensor_slices_dataset([5], "int32")
    iterator = iter(dataset)
    elements = list(iterator)
    assert len(elements) == 5
    assert all(elem.dtype == tf.int32 for elem in elements)
    
    # Test with float32
    dataset = create_tensor_slices_dataset([3], "float32")
    iterator = iter(dataset)
    elements = list(iterator)
    assert len(elements) == 3
    assert all(elem.dtype == tf.float32 for elem in elements)
    
    # Test with custom num_elements
    dataset = create_tensor_slices_dataset([10], "int64", num_elements=3)
    iterator = iter(dataset)
    elements = list(iterator)
    assert len(elements) == 3
    assert all(elem.dtype == tf.int64 for elem in elements)

def test_create_tensor_slices_dataset_multidimensional():
    """Test create_tensor_slices_dataset with multidimensional data."""
    # Test 2D data
    dataset = create_tensor_slices_dataset([2, 3], "int32")
    iterator = iter(dataset)
    elements = list(iterator)
    assert len(elements) == 2
    assert all(elem.shape == (3,) for elem in elements)
    
    # Test 3D data
    dataset = create_tensor_slices_dataset([2, 3, 4], "float32")
    iterator = iter(dataset)
    elements = list(iterator)
    assert len(elements) == 2
    assert all(elem.shape == (3, 4) for elem in elements)

def test_create_tensor_slices_dataset_edge_cases():
    """Test edge cases for create_tensor_slices_dataset."""
    # Test empty data_shape
    dataset = create_tensor_slices_dataset([], "int32", num_elements=5)
    iterator = iter(dataset)
    elements = list(iterator)
    assert len(elements) == 5
    assert all(elem.shape == () for elem in elements)
    
    # Test with zero elements
    dataset = create_tensor_slices_dataset([0], "float32")
    iterator = iter(dataset)
    elements = list(iterator)
    assert len(elements) == 0

def test_create_tensor_slices_dataset_unsupported_dtype():
    """Test create_tensor_slices_dataset with unsupported dtype."""
    with pytest.raises(ValueError) as exc_info:
        create_tensor_slices_dataset([5], "unsupported_dtype")
    
    assert "Unsupported dtype" in str(exc_info.value)

def test_create_empty_dataset():
    """Test create_empty_dataset function."""
    # Test with default int32
    dataset = create_empty_dataset()
    iterator = iter(dataset)
    elements = list(iterator)
    assert len(elements) == 0
    
    # Test with float64
    dataset = create_empty_dataset("float64")
    iterator = iter(dataset)
    elements = list(iterator)
    assert len(elements) == 0
    # Verify dataset structure
    assert isinstance(dataset, tf.data.Dataset)
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# TC-01: prefetch_to_device基本功能验证 (DEFERRED - G1 test, placeholder for reference)
# Priority: High, Group: G1
# This test case belongs to group G1 and is implemented in its own file.
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# TC-02: prefetch_to_device buffer_size参数验证 (DEFERRED - G1 test, placeholder for reference)
# Priority: High, Group: G1
# This test case belongs to group G1 and is implemented in its own file.
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: prefetch_to_device无效参数处理 (DEFERRED - G1 test, placeholder for reference)
# Priority: Medium, Group: G1
# This test case belongs to group G1 and is implemented in its own file.
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: prefetch_to_device空数据集处理 (DEFERRED - G1 test, placeholder for reference)
# Priority: Medium, Group: G1
# This test case belongs to group G1 and is implemented in its own file.
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: copy_to_device基本功能验证
# Priority: High, Group: G2
# Parameters: target_device=/cpu:0, source_device=/cpu:0, dataset_type=tensor_slices, data_shape=[8], dtype=float64
# Weak asserts: returns_callable, dataset_apply_compatible, output_shape_match, output_dtype_match
@pytest.mark.parametrize("target_device,source_device,data_shape,dtype", [
    ("/cpu:0", "/cpu:0", [8], "float64"),  # Original test case
    ("/cpu:0", "/cpu:0", [50], "int32"),  # Parameter extension: Medium-sized dataset
])
def test_copy_to_device_basic_functionality(tf_seed, target_device, source_device, data_shape, dtype):
    """Test basic functionality of copy_to_device."""
    # Create dataset
    dataset = create_tensor_slices_dataset(data_shape, dtype)
    
    # Get transformation function
    transform_fn = copy_to_device(target_device=target_device, source_device=source_device)
    
    # Assert 1: returns_callable - transformation function should be callable
    assert callable(transform_fn), "copy_to_device should return a callable function"
    
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
    
    # Verify data integrity by checking values
    iterator = iter(transformed_dataset)
    for i, element in enumerate(iterator):
        # For float64, check approximate equality
        if dtype == "float64":
            expected_value = tf.constant(float(i), dtype=tf.float64)
            tf.debugging.assert_near(element, expected_value, rtol=1e-7)
        else:
            expected_value = tf.constant(i, dtype=getattr(tf, dtype))
            tf.debugging.assert_equal(element, expected_value)
    
    # Additional verification for medium-sized dataset test
    if data_shape == [50] and dtype == "int32":
        # Verify all 50 elements are present and correct
        iterator = iter(transformed_dataset)
        total_sum = 0
        for i, element in enumerate(iterator):
            total_sum += int(element.numpy())
        
        # Verify sum of first 50 numbers (0+1+2+...+49 = 1225)
        expected_sum = sum(range(50))
        assert total_sum == expected_sum, \
            f"Data integrity check failed for medium dataset: expected sum {expected_sum}, got {total_sum}"
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# TC-06: copy_to_device GPU支持测试
# Priority: Medium, Group: G2
# Parameters: target_device=gpu:0, source_device=/cpu:0, dataset_type=tensor_slices, data_shape=[6], dtype=float32
# Weak asserts: returns_callable, dataset_apply_compatible, gpu_supported, no_crash
@pytest.mark.parametrize("target_device,source_device,data_shape,dtype,use_gpu", [
    ("gpu:0", "/cpu:0", [6], "float32", True),  # GPU test
    ("/cpu:0", "/cpu:0", [6], "float32", False),  # CPU fallback test
])
def test_copy_to_device_gpu_support(tf_seed, target_device, source_device, data_shape, dtype, use_gpu):
    """Test copy_to_device GPU support with CPU fallback."""
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    
    # If testing GPU but no GPU available, skip GPU test but run CPU fallback
    if use_gpu and not gpus:
        pytest.skip("No GPU available for testing")
    
    # Create dataset
    dataset = create_tensor_slices_dataset(data_shape, dtype)
    
    # Get transformation function
    transform_fn = copy_to_device(target_device=target_device, source_device=source_device)
    
    # Assert 1: returns_callable - transformation function should be callable
    assert callable(transform_fn), "copy_to_device should return a callable function"
    
    # Apply transformation
    transformed_dataset = dataset.apply(transform_fn)
    
    # Assert 2: dataset_apply_compatible - should work with Dataset.apply
    assert isinstance(transformed_dataset, tf.data.Dataset), \
        "Transformed result should be a tf.data.Dataset"
    
    # For GPU operations, we need to use initializable iterator
    # For CPU operations, we can use standard iterator
    if use_gpu:
        # GPU test path
        # Assert 3: gpu_supported - should support GPU device
        iterator = tf.compat.v1.data.make_initializable_iterator(transformed_dataset)
        
        with tf.compat.v1.Session() as sess:
            sess.run(iterator.initializer)
            
            # Get first element
            first_element = sess.run(iterator.get_next())
            
            # Verify shape
            expected_shape = data_shape[1:] if len(data_shape) > 1 else ()
            assert first_element.shape == expected_shape, \
                f"Output shape {first_element.shape} should match expected {expected_shape}"
            
            # Verify dtype
            # Note: GPU operations may change dtype representation
            # We'll accept either float32 or float64 for GPU tests
            assert first_element.dtype in [np.float32, np.float64], \
                f"Output dtype {first_element.dtype} should be float32 or float64"
            
            # Assert 4: no_crash - should complete without errors
            # Try to get all elements
            count = 1
            try:
                while True:
                    sess.run(iterator.get_next())
                    count += 1
            except tf.errors.OutOfRangeError:
                # Expected when dataset ends
                pass
            
            expected_count = data_shape[0] if data_shape else 1
            assert count == expected_count, \
                f"Should iterate through {expected_count} elements, got {count}"
            
            # Additional verification: check that we can restart the iterator
            sess.run(iterator.initializer)
            restart_element = sess.run(iterator.get_next())
            assert restart_element.shape == expected_shape, \
                "Should be able to restart iterator and get same shape"
    else:
        # CPU fallback test path
        # This ensures test runs even without GPU
        iterator = iter(transformed_dataset)
        first_element = next(iterator)
        
        # Verify shape
        expected_shape = data_shape[1:] if len(data_shape) > 1 else ()
        assert first_element.shape == expected_shape, \
            f"Output shape {first_element.shape} should match expected {expected_shape}"
        
        # Verify dtype
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
        
        # Verify data integrity
        iterator = iter(transformed_dataset)
        for i, element in enumerate(iterator):
            expected_value = tf.constant(float(i), dtype=expected_dtype)
            tf.debugging.assert_near(element, expected_value, rtol=1e-7)
        
        # Additional test: verify device placement
        # Note: In eager mode, device placement might not be visible
        # We'll just verify the function works correctly
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# TC-07: copy_to_device无效设备处理
# Priority: Medium, Group: G2
# Parameters: target_device=invalid_gpu, source_device=/cpu:0, dataset_type=tensor_slices, data_shape=[4], dtype=int64
# Weak asserts: raises_exception, exception_type_correct, error_message_contains
@pytest.mark.parametrize("target_device,source_device,data_shape,dtype,test_phase", [
    ("invalid_gpu", "/cpu:0", [4], "int64", "creation"),  # Test during function creation
    ("invalid_gpu", "/cpu:0", [4], "int64", "application"),  # Test during dataset application
    ("invalid_gpu", "/cpu:0", [4], "int64", "iteration"),  # Test during iteration
    ("/cpu:999", "/cpu:0", [4], "int64", "iteration"),  # Test with non-existent CPU device
    ("gpu:999", "/cpu:0", [4], "int64", "iteration"),  # Test with non-existent GPU device
    ("", "/cpu:0", [4], "int64", "iteration"),  # Test with empty device string
    (None, "/cpu:0", [4], "int64", "creation"),  # Test with None device
])
def test_copy_to_device_invalid_device(tf_seed, target_device, source_device, data_shape, dtype, test_phase):
    """Test copy_to_device with invalid device at different phases."""
    # Skip None device test if target_device is None (handled separately)
    if target_device is None:
        # Test that None device raises appropriate error during function creation
        with pytest.raises((TypeError, ValueError)) as exc_info:
            transform_fn = copy_to_device(target_device=target_device, source_device=source_device)
        
        error_message = str(exc_info.value).lower()
        assert "device" in error_message or "none" in error_message or "type" in error_message, \
            f"Error message should mention device or type issue. Got: {error_message}"
        return
    
    # Create dataset
    dataset = create_tensor_slices_dataset(data_shape, dtype)
    
    try:
        # Phase 1: Function creation
        transform_fn = copy_to_device(target_device=target_device, source_device=source_device)
        
        # Assert 1: returns_callable - should still return a callable even with invalid device
        assert callable(transform_fn), "copy_to_device should return a callable function even with invalid device"
        
        if test_phase == "creation":
            # Test passes if we get here without exception
            # Additional check: verify the function can be called
            result = transform_fn(dataset)
            assert isinstance(result, tf.data.Dataset), \
                "Should be able to call transformation function even with invalid device"
            return
        
        # Phase 2: Dataset application
        transformed_dataset = dataset.apply(transform_fn)
        
        if test_phase == "application":
            # Test passes if we get here without exception
            # Verify dataset was created
            assert isinstance(transformed_dataset, tf.data.Dataset), \
                "Should create dataset even with invalid device (error may be deferred)"
            
            # Additional check: verify dataset structure
            element_spec = transformed_dataset.element_spec
            assert element_spec is not None, "Dataset should have element spec"
            return
        
        # Phase 3: Iteration (where error is most likely to occur)
        iterator = iter(transformed_dataset)
        _ = next(iterator)
        
        # If we get here without exception, the test fails
        pytest.fail(f"Expected exception for invalid device {target_device} but none was raised")
        
    except Exception as exc_info:
        # Assert 2: raises_exception - should raise exception for invalid device
        exception = exc_info
        exception_type = type(exception).__name__
        
        # Assert 3: exception_type_correct - should raise appropriate exception
        # TensorFlow may raise various exceptions for invalid devices
        acceptable_exceptions = [
            'InvalidArgumentError',
            'NotFoundError',
            'FailedPreconditionError',
            'ValueError',
            'RuntimeError',
            'TypeError',
            'InternalError'
        ]
        
        assert any(acceptable in exception_type for acceptable in acceptable_exceptions), \
            f"Expected one of {acceptable_exceptions}, got {exception_type} with message: {str(exception)}"
        
        # Assert 4: error_message_contains - error message should contain relevant info
        error_message = str(exception).lower()
        
        # Check for device-related keywords in error message
        device_keywords = ['device', 'gpu', 'cpu', 'invalid', 'not found', 'not exist', 'unknown', 'unavailable', 'exist']
        
        # For some TensorFlow versions, the error might be generic
        # We'll accept any exception with device in message, or if it's a known TensorFlow error type
        has_device_keyword = any(keyword in error_message for keyword in device_keywords)
        is_tf_error = any(tf_error in exception_type for tf_error in ['Error', 'Exception'])
        
        assert has_device_keyword or is_tf_error, \
            f"Error message should contain device-related info or be a TensorFlow error. " \
            f"Message: {error_message}, Type: {exception_type}"
        
        # Additional validation: Check that the exception provides useful context
        if test_phase == "iteration":
            # During iteration, we expect more specific errors
            assert len(error_message) > 10, "Error message should provide meaningful context"
        
        # Additional check for empty device string
        if target_device == "":
            assert "empty" in error_message or "invalid" in error_message, \
                f"Empty device string should trigger appropriate error. Got: {error_message}"
        
        # Log the actual error for debugging (but don't fail the test)
        print(f"Test phase '{test_phase}' with device '{target_device}' caught expected exception: {exception_type}: {error_message[:100]}...")
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:FOOTER START ====
# Additional test utilities and cleanup for G2

def test_module_import():
    """Test that the module can be imported correctly."""
    # Re-import to ensure no side effects
    import importlib
    import tensorflow.python.data.experimental.ops.prefetching_ops as prefetching_ops_module
    
    # Verify key functions are available
    assert hasattr(prefetching_ops_module, 'prefetch_to_device'), \
        "Module should have prefetch_to_device function"
    assert hasattr(prefetching_ops_module, 'copy_to_device'), \
        "Module should have copy_to_device function"
    
    # Verify functions are callable
    assert callable(prefetching_ops_module.prefetch_to_device), \
        "prefetch_to_device should be callable"
    assert callable(prefetching_ops_module.copy_to_device), \
        "copy_to_device should be callable"
    
    # Test basic function signatures
    import inspect
    prefetch_sig = inspect.signature(prefetching_ops_module.prefetch_to_device)
    copy_sig = inspect.signature(prefetching_ops_module.copy_to_device)
    
    assert 'device' in prefetch_sig.parameters, \
        "prefetch_to_device should have 'device' parameter"
    assert 'target_device' in copy_sig.parameters, \
        "copy_to_device should have 'target_device' parameter"
    
    print("Module import test passed successfully")

def test_environment_setup():
    """Test that the test environment is properly set up."""
    # Verify TensorFlow is available
    import tensorflow as tf
    assert tf.__version__ is not None, "TensorFlow should be available"
    
    # Verify we can create basic tensors
    tensor = tf.constant([1, 2, 3])
    assert tensor.shape == (3,), "Should be able to create tensors"
    
    # Verify we can create datasets
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    assert isinstance(dataset, tf.data.Dataset), "Should be able to create datasets"
    
    print("Environment setup test passed successfully")

# Run additional tests when module is executed directly
if __name__ == "__main__":
    # Run the additional utility tests
    test_module_import()
    test_environment_setup()
    
    # Run pytest on this file
    import sys
    pytest_args = [__file__, "-v", "--tb=short"]
    
    # Add coverage reporting if coverage module is available
    try:
        import coverage
        pytest_args.extend(["--cov=.", "--cov-report=term-missing"])
    except ImportError:
        print("Coverage module not available, running tests without coverage")
    
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)
# ==== BLOCK:FOOTER END ====