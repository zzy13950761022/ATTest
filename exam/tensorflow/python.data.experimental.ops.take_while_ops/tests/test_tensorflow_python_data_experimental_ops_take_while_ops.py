import warnings
import pytest
import tensorflow as tf
from unittest.mock import Mock, patch, call
from tensorflow.python.data.experimental.ops.take_while_ops import take_while

# ==== BLOCK:HEADER START ====
# Test setup and common imports for take_while function tests

# Common setup for deprecation warning handling
def setup_module():
    """Module-level setup"""
    warnings.filterwarnings("always", category=DeprecationWarning)

def teardown_module():
    """Module-level teardown"""
    warnings.resetwarnings()

# Common test fixtures and helpers
def create_mock_predicate(return_value=tf.constant(True)):
    """Helper to create a mock predicate function"""
    return Mock(return_value=return_value)

def create_mock_dataset():
    """Helper to create a mock dataset with take_while method"""
    dataset = Mock(spec=tf.data.Dataset)
    dataset.take_while = Mock()
    return dataset
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# TC-01: 函数返回类型验证
# SMOKE_SET - G1
def test_take_while_function_type_and_deprecation():
    """Test that take_while returns a callable function and triggers deprecation warning"""
    # Arrange
    predicate = lambda x: tf.constant(True)
    
    # Act - capture deprecation warning via warnings module
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        
        # Call the function - deprecation warning should be triggered
        result = take_while(predicate)
        
        # Assert: deprecation warning was issued
        # TensorFlow's deprecation decorator uses warnings.warn
        assert len(w) > 0, "At least one warning should be issued"
        
        # Check for deprecation warnings
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0, "Should have at least one DeprecationWarning"
        
        # Check warning message contains deprecation info
        warning_msg = str(deprecation_warnings[0].message).lower()
        assert "deprecated" in warning_msg or "future version" in warning_msg, \
            f"Warning message should indicate deprecation. Got: {warning_msg}"
        
        # Check for tf.data.Dataset.take_while suggestion
        assert "tf.data.dataset.take_while" in warning_msg.lower() or "use tf.data.dataset.take_while" in warning_msg.lower(), \
            f"Warning should suggest using tf.data.Dataset.take_while. Got: {warning_msg}"
    
    # Assert: result is callable
    assert callable(result), "take_while should return a callable function"
    
    # Assert: result has correct signature (accepts dataset parameter)
    try:
        import inspect
        sig = inspect.signature(result)
        params = list(sig.parameters.keys())
        assert len(params) == 1, f"Returned function should accept exactly one parameter, got {len(params)}: {params}"
        # Parameter name may vary, but should accept one argument
    except (AttributeError, ValueError):
        # inspect.signature may not work on all callables, but callable check is sufficient
        pass
    
    # Additional test: verify the returned function works with a dataset
    dataset = tf.data.Dataset.range(5)
    transformed = result(dataset)
    assert isinstance(transformed, tf.data.Dataset), \
        "Returned function should transform dataset to dataset"
    
    # Verify the transformation actually works
    result_list = list(transformed.as_numpy_iterator())
    assert len(result_list) == 5, "Should get all 5 elements from range dataset with always-true predicate"
    assert result_list == [0, 1, 2, 3, 4], "Should get correct values from range dataset"
    
    # Test with a predicate that stops early
    def stop_at_3(x):
        return tf.constant(x < 3)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        stop_func = take_while(stop_at_3)
        stopped_dataset = stop_func(dataset)
        stopped_list = list(stopped_dataset.as_numpy_iterator())
        assert len(stopped_list) == 3, "Should stop at 3"
        assert stopped_list == [0, 1, 2], "Should get values 0, 1, 2"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# TC-02: 转换函数正确包装predicate
# SMOKE_SET - G1
def test_take_while_wraps_predicate_correctly():
    """Test that the transformation function correctly wraps the predicate"""
    # Arrange
    mock_predicate = Mock(return_value=tf.constant(True))
    mock_dataset = Mock(spec=tf.data.Dataset)
    mock_dataset.take_while = Mock()
    
    # Act - capture deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        transform_func = take_while(mock_predicate)
    
    # Assert: deprecation warning was issued
    assert len(w) > 0, "Should issue deprecation warning"
    deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
    assert len(deprecation_warnings) > 0, "Should have DeprecationWarning"
    
    # Act: apply transformation to dataset
    result = transform_func(mock_dataset)
    
    # Assert: transform_func called dataset.take_while
    mock_dataset.take_while.assert_called_once()
    
    # Assert: predicate was passed correctly as keyword argument
    call_args = mock_dataset.take_while.call_args
    assert "predicate" in call_args.kwargs, "predicate should be passed as keyword argument"
    assert call_args.kwargs["predicate"] is mock_predicate, "Predicate function should be passed unchanged"
    
    # Assert: no positional arguments
    assert len(call_args.args) == 0, "take_while should be called with only keyword arguments"
    
    # Assert: result is what dataset.take_while returns
    assert result is mock_dataset.take_while.return_value, "Transform should return dataset.take_while result"
    
    # Additional test: verify predicate is called with correct arguments during iteration
    # Create a real dataset with a mock predicate
    call_log = []
    
    def logging_predicate(x):
        call_log.append(x.numpy())
        return tf.constant(True)
    
    real_dataset = tf.data.Dataset.range(3)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        real_transform = take_while(logging_predicate)
        real_result = real_transform(real_dataset)
        
        # Iterate through the dataset
        result_list = list(real_result.as_numpy_iterator())
    
    # Verify predicate was called with correct values
    assert len(call_log) == 3, "Predicate should be called 3 times for range(3)"
    assert call_log == [0, 1, 2], "Predicate should receive values 0, 1, 2"
    assert result_list == [0, 1, 2], "Should get all values from range(3)"
    
    # Test with predicate that stops early
    early_stop_log = []
    
    def stop_at_1(x):
        early_stop_log.append(x.numpy())
        return tf.constant(x < 1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        stop_transform = take_while(stop_at_1)
        stop_result = stop_transform(real_dataset)
        stop_list = list(stop_result.as_numpy_iterator())
    
    assert len(early_stop_log) == 2, "Predicate should be called twice (for 0 and 1)"
    assert early_stop_log == [0, 1], "Predicate should receive values 0 and 1"
    assert stop_list == [0], "Should stop after first element"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: predicate返回False时停止迭代
# SMOKE_SET - G1
def test_take_while_stops_when_predicate_returns_false():
    """Test that iteration stops when predicate returns False"""
    # Arrange: create a dataset and predicate that returns False immediately
    dataset = tf.data.Dataset.range(10)
    
    # Counter to track how many times predicate is called
    call_count = 0
    
    def predicate_always_false(x):
        nonlocal call_count
        call_count += 1
        return tf.constant(False)
    
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        transform_func = take_while(predicate_always_false)
        result_dataset = transform_func(dataset)
        
        # Collect results from the transformed dataset
        result_list = list(result_dataset.as_numpy_iterator())
    
    # Assert: predicate was called at least once
    assert call_count >= 1, "Predicate should be called at least once"
    
    # Assert: result dataset is empty (since predicate returns False immediately)
    assert len(result_list) == 0, "Dataset should be empty when predicate returns False immediately"
    
    # Additional check: verify the dataset transformation works
    assert isinstance(result_dataset, tf.data.Dataset), "Result should be a tf.data.Dataset"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: predicate返回标量布尔张量
# DEFERRED_SET - G1
def test_take_while_with_tensorflow_bool_scalar():
    """Test predicate returning scalar boolean tensor"""
    # Arrange: create dataset with tensor slices
    dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
    
    # Create predicate that returns scalar boolean tensor
    def tensor_bool_predicate(x):
        # x is a tensor of shape (2,)
        # Return scalar boolean based on sum
        sum_val = tf.reduce_sum(x)
        return tf.constant(sum_val < 5, dtype=tf.bool)  # Scalar boolean tensor
    
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        transform_func = take_while(tensor_bool_predicate)
        result_dataset = transform_func(dataset)
        
        # Collect results
        result_list = list(result_dataset.as_numpy_iterator())
    
    # Assert: no error raised
    # The predicate returns proper scalar boolean tensor
    
    # Assert: correct filtering based on predicate
    # First element [1, 2] sum = 3 < 5 → True → include
    # Second element [3, 4] sum = 7 < 5 → False → stop
    assert len(result_list) == 1, "Should include only first element"
    assert result_list[0].tolist() == [1, 2], "Should get first tensor slice"
    
    # Test with always-true tensor boolean predicate
    def always_true_predicate(x):
        return tf.constant(True, dtype=tf.bool)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        true_transform = take_while(always_true_predicate)
        true_result = true_transform(dataset)
        true_list = list(true_result.as_numpy_iterator())
    
    assert len(true_list) == 2, "Should include all elements with always-true predicate"
    assert true_list[0].tolist() == [1, 2] and true_list[1].tolist() == [3, 4]
    
    # Test with tensor operations in predicate
    def complex_tensor_predicate(x):
        # Use tensor operations, still returning scalar boolean
        norm = tf.norm(x)
        return tf.constant(norm < 3.0, dtype=tf.bool)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        complex_transform = take_while(complex_tensor_predicate)
        complex_result = complex_transform(dataset)
        complex_list = list(complex_result.as_numpy_iterator())
    
    # [1, 2] norm ≈ 2.236 < 3.0 → True
    # [3, 4] norm = 5.0 < 3.0 → False
    assert len(complex_list) == 1, "Should stop at second element"
    assert complex_list[0].tolist() == [1, 2]
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: predicate参数非函数类型异常
# SMOKE_SET - G2
@pytest.mark.parametrize("invalid_predicate,expected_error", [
    (None, (TypeError, ValueError)),  # TensorFlow may raise TypeError or ValueError
    ("not_a_function", (TypeError, ValueError)),
    (123, (TypeError, ValueError)),
])
def test_take_while_invalid_predicate_type(invalid_predicate, expected_error):
    """Test that non-function predicate arguments raise appropriate error"""
    # Note: take_while itself doesn't validate the predicate immediately.
    # The validation happens when the returned function is called with a dataset.
    
    # Act & Assert: should raise error when trying to use the transformation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        
        # Step 1: take_while should succeed and return a callable
        transform_func = take_while(invalid_predicate)
        assert callable(transform_func), "take_while should return a callable function"
        
        # Step 2: The error should be raised when applying to a dataset
        dataset = tf.data.Dataset.range(5)
        
        # The error might be raised immediately when calling transform_func
        # or during iteration of the transformed dataset
        # We use a more flexible assertion that accepts either case
        try:
            transformed = transform_func(dataset)
            # If no error during transformation, try to iterate
            # The error should be raised during iteration
            with pytest.raises(expected_error) as exc_info:
                list(transformed.as_numpy_iterator())
        except expected_error as exc_info:
            # Error was raised during transformation, which is also acceptable
            pass
    
    # Note: We accept either TypeError or ValueError because TensorFlow's
    # internal validation may raise either depending on the exact implementation
    # and TensorFlow version. The important thing is that an error is raised.
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# TC-06: predicate返回非布尔类型异常
# DEFERRED_SET - G2
@pytest.mark.parametrize("predicate_type,dataset_args,expected_error", [
    ("returns_integer", {"stop": 5}, (ValueError, TypeError)),
    ("returns_string", {"stop": 5}, (ValueError, TypeError)),
])
def test_take_while_predicate_returns_non_boolean(predicate_type, dataset_args, expected_error):
    """Test predicate returning non-boolean type raises error"""
    # Arrange: create dataset
    dataset = tf.data.Dataset.range(dataset_args["stop"])
    
    # Create predicate based on type
    if predicate_type == "returns_integer":
        def predicate(x):
            return tf.constant(42, dtype=tf.int32)  # Returns integer instead of boolean
    elif predicate_type == "returns_string":
        def predicate(x):
            return tf.constant("not_a_boolean", dtype=tf.string)  # Returns string instead of boolean
    else:
        pytest.fail(f"Unknown predicate type: {predicate_type}")
    
    # Act & Assert: should raise error when predicate returns non-boolean
    # Note: The error is raised immediately when calling transform_func(dataset)
    # because TensorFlow validates the predicate's return type at construction time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        transform_func = take_while(predicate)
        
        # The error should be raised immediately when applying the transformation
        # Accept either ValueError or TypeError as TensorFlow may raise either
        with pytest.raises(expected_error) as exc_info:
            result_dataset = transform_func(dataset)
            # Try to iterate to trigger validation if not already triggered
            list(result_dataset.as_numpy_iterator())
        
        # Verify the error message contains relevant information
        error_msg = str(exc_info.value).lower()
        # Check for predicate-related keywords
        predicate_keywords = ["predicate", "function", "return"]
        bool_keywords = ["bool", "boolean", "scalar", "tensor"]
        
        # At least one predicate keyword should be present
        has_predicate_keyword = any(keyword in error_msg for keyword in predicate_keywords)
        has_bool_keyword = any(keyword in error_msg for keyword in bool_keywords)
        
        # Log the actual error message for debugging
        print(f"Error message for {predicate_type}: {error_msg}")
        
        # Assert that error message is informative
        assert has_predicate_keyword or has_bool_keyword, \
            f"Error message should mention predicate or boolean type. Got: {error_msg}"
    
    # Additional test: verify that a valid boolean predicate works
    def valid_predicate(x):
        return tf.constant(True, dtype=tf.bool)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        valid_transform = take_while(valid_predicate)
        valid_result = valid_transform(dataset)
        # Should not raise error
        list(valid_result.as_numpy_iterator())
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# TC-07: predicate返回非标量布尔张量
# DEFERRED_SET - G2
def test_take_while_predicate_returns_non_scalar_bool_tensor():
    """Test predicate returning non-scalar boolean tensor raises error (deferred)"""
    # Deferred implementation
    pytest.skip("Deferred test case - will be implemented in later rounds")
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# TC-08: 空数据集边界处理
# DEFERRED_SET - G2
def test_take_while_with_empty_dataset():
    """Test take_while with empty dataset (deferred)"""
    # Deferred implementation
    pytest.skip("Deferred test case - will be implemented in later rounds")
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Test execution entry point
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
# ==== BLOCK:FOOTER END ====