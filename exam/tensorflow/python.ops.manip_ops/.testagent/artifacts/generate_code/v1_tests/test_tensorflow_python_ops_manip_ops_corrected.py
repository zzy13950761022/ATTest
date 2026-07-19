"""
Test cases for tensorflow.python.ops.manip_ops.roll function.
"""
import numpy as np
import tensorflow as tf
import pytest

# ==== BLOCK:HEADER START ====
# Helper functions
def numpy_roll_reference(input_array, shift, axis):
    """Reference implementation using numpy.roll for validation."""
    if isinstance(shift, (list, tuple, np.ndarray)):
        if isinstance(axis, (list, tuple, np.ndarray)):
            # Multiple axes case
            result = input_array.copy()
            for s, a in zip(shift, axis):
                result = np.roll(result, shift=s, axis=a)
            return result
        else:
            # Single axis with multiple shifts (should be same axis)
            result = input_array.copy()
            for s in shift:
                result = np.roll(result, shift=s, axis=axis)
            return result
    else:
        # Single shift, single axis
        return np.roll(input_array, shift=shift, axis=axis)

def create_test_tensor(shape, dtype):
    """Create test tensor with deterministic values."""
    np.random.seed(42)
    if dtype in [tf.float32, tf.float64]:
        return tf.constant(np.random.randn(*shape).astype(dtype.as_numpy_dtype))
    elif dtype in [tf.int32, tf.int64]:
        return tf.constant(np.random.randint(0, 100, size=shape, dtype=dtype.as_numpy_dtype))
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize(
    "input_shape,input_dtype,shift,axis,expected_pattern",
    [
        # Base case from test plan
        ([5], tf.float32, 2, 0, "正向滚动2位"),
        # Parameter extensions
        ([5], tf.int64, -3, 0, "负向滚动3位"),
        ([7], tf.float64, 10, 0, "大偏移量滚动（超过维度长度）"),
    ]
)
def test_1d_tensor_basic_roll(input_shape, input_dtype, shift, axis, expected_pattern):
    """Test basic roll operations on 1D tensors."""
    # Create test tensor
    input_tensor = create_test_tensor(input_shape, input_dtype)
    
    # Convert to numpy for reference calculation
    input_np = input_tensor.numpy()
    
    # Calculate expected result using numpy reference
    expected_np = numpy_roll_reference(input_np, shift, axis)
    
    # Perform TensorFlow roll operation
    from tensorflow.python.ops import manip_ops
    result_tf = manip_ops.roll(input_tensor, shift=shift, axis=axis)
    
    # Convert to numpy for comparison
    result_np = result_tf.numpy()
    
    # Weak assertions (shape and dtype match)
    assert result_tf.shape == input_tensor.shape, (
        f"Shape mismatch: expected {input_tensor.shape}, got {result_tf.shape}"
    )
    assert result_tf.dtype == input_tensor.dtype, (
        f"Dtype mismatch: expected {input_tensor.dtype}, got {result_tf.dtype}"
    )
    
    # Basic roll correctness (element-wise comparison)
    np.testing.assert_array_equal(result_np, expected_np, 
        f"Roll operation incorrect for pattern: {expected_pattern}")
    
    # Additional validation for large shifts
    if isinstance(shift, (int, float)) and abs(shift) >= input_shape[0]:
        # For large shifts, verify cyclic property
        # After shifting by dimension length, tensor should be unchanged
        shift_mod = shift % input_shape[0]
        expected_cyclic = numpy_roll_reference(input_np, shift_mod, axis)
        np.testing.assert_array_equal(result_np, expected_cyclic,
            f"Large shift {shift} should be equivalent to shift {shift_mod}")
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize(
    "input_shape,input_dtype,shift,axis,expected_pattern",
    [
        # Base case from test plan
        ([3, 4], tf.int32, [1, -1], [0, 1], "行正向滚动1位，列负向滚动1位"),
        # Parameter extension
        ([2, 3, 4], tf.float32, [0, 1, -2], [0, 1, 2], "三维张量多轴滚动"),
    ]
)
def test_multi_axis_roll(input_shape, input_dtype, shift, axis, expected_pattern):
    """Test roll operations on multi-dimensional tensors with multiple axes."""
    # Create test tensor
    input_tensor = create_test_tensor(input_shape, input_dtype)
    
    # Convert to numpy for reference calculation
    input_np = input_tensor.numpy()
    
    # Calculate expected result using numpy reference
    expected_np = numpy_roll_reference(input_np, shift, axis)
    
    # Perform TensorFlow roll operation
    from tensorflow.python.ops import manip_ops
    result_tf = manip_ops.roll(input_tensor, shift=shift, axis=axis)
    
    # Convert to numpy for comparison
    result_np = result_tf.numpy()
    
    # Weak assertions (shape and dtype match)
    assert result_tf.shape == input_tensor.shape, (
        f"Shape mismatch: expected {input_tensor.shape}, got {result_tf.shape}"
    )
    assert result_tf.dtype == input_tensor.dtype, (
        f"Dtype mismatch: expected {input_tensor.dtype}, got {result_tf.dtype}"
    )
    
    # Multi-axis correctness (element-wise comparison)
    np.testing.assert_array_equal(result_np, expected_np,
        f"Multi-axis roll operation incorrect for pattern: {expected_pattern}")
    
    # Additional validation for independent axis operations
    # For 2D case, verify that row and column shifts are independent
    if len(input_shape) == 2 and len(shift) == 2 and len(axis) == 2:
        # Apply shifts sequentially and compare
        # First shift along axis[0], then axis[1]
        intermediate = np.roll(input_np, shift=shift[0], axis=axis[0])
        expected_sequential = np.roll(intermediate, shift=shift[1], axis=axis[1])
        np.testing.assert_array_equal(result_np, expected_sequential,
            f"Multi-axis shifts should be independent and commutative")
        
        # Verify commutative property (order shouldn't matter for different axes)
        # First shift along axis[1], then axis[0]
        intermediate2 = np.roll(input_np, shift=shift[1], axis=axis[1])
        expected_commutative = np.roll(intermediate2, shift=shift[0], axis=axis[0])
        np.testing.assert_array_equal(result_np, expected_commutative,
            f"Multi-axis shifts should be commutative for different axes")
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize(
    "input_shape,input_dtype,shift,axis,expected_pattern",
    [
        # Base case from test plan
        ([4], tf.float64, [2, -1], [0, 0], "同轴偏移累加（净偏移1位）"),
    ]
)
def test_same_axis_cumulative_roll(input_shape, input_dtype, shift, axis, expected_pattern):
    """Test roll operations with multiple shifts on the same axis."""
    # Create test tensor
    input_tensor = create_test_tensor(input_shape, input_dtype)
    
    # Convert to numpy for reference calculation
    input_np = input_tensor.numpy()
    
    # Calculate expected result using numpy reference
    # For same axis multiple shifts, numpy applies them sequentially
    expected_np = numpy_roll_reference(input_np, shift, axis)
    
    # Perform TensorFlow roll operation
    from tensorflow.python.ops import manip_ops
    result_tf = manip_ops.roll(input_tensor, shift=shift, axis=axis)
    
    # Convert to numpy for comparison
    result_np = result_tf.numpy()
    
    # Weak assertions (shape and dtype match)
    assert result_tf.shape == input_tensor.shape, (
        f"Shape mismatch: expected {input_tensor.shape}, got {result_tf.shape}"
    )
    assert result_tf.dtype == input_tensor.dtype, (
        f"Dtype mismatch: expected {input_tensor.dtype}, got {result_tf.dtype}"
    )
    
    # Cumulative shift correctness (element-wise comparison)
    np.testing.assert_array_equal(result_np, expected_np,
        f"Cumulative roll operation incorrect for pattern: {expected_pattern}")
    
    # Additional validation for cumulative property
    # Calculate net shift (sum of all shifts on same axis)
    net_shift = sum(shift)
    
    # Apply single shift with net value
    expected_net = np.roll(input_np, shift=net_shift, axis=axis[0])
    
    # Verify that multiple shifts on same axis are equivalent to single net shift
    np.testing.assert_array_equal(result_np, expected_net,
        f"Multiple shifts on same axis should be equivalent to net shift {net_shift}")
    
    # Verify order independence for same axis
    # Shifts on same axis should be commutative
    if len(shift) == 2:
        # Apply shifts in reverse order
        reverse_result = np.roll(input_np, shift=shift[1], axis=axis[1])
        reverse_result = np.roll(reverse_result, shift=shift[0], axis=axis[0])
        np.testing.assert_array_equal(result_np, reverse_result,
            f"Shifts on same axis should be commutative")
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
@pytest.mark.parametrize(
    "input_shape,input_dtype,shift,axis,expected_pattern",
    [
        # Base case from test plan
        ([0], tf.int32, 0, 0, "空张量滚动无变化"),
        # Parameter extension
        ([], tf.float32, 0, 0, "标量张量滚动无变化"),
    ]
)
def test_boundary_conditions_empty_scalar(input_shape, input_dtype, shift, axis, expected_pattern):
    """Test roll operations on empty tensors and scalars."""
    # Create test tensor
    input_tensor = create_test_tensor(input_shape, input_dtype)
    
    # Convert to numpy for reference calculation
    input_np = input_tensor.numpy()
    
    # Calculate expected result using numpy reference
    expected_np = numpy_roll_reference(input_np, shift, axis)
    
    # Perform TensorFlow roll operation
    from tensorflow.python.ops import manip_ops
    result_tf = manip_ops.roll(input_tensor, shift=shift, axis=axis)
    
    # Convert to numpy for comparison
    result_np = result_tf.numpy()
    
    # Weak assertions (shape and dtype match)
    assert result_tf.shape == input_tensor.shape, (
        f"Shape mismatch: expected {input_tensor.shape}, got {result_tf.shape}"
    )
    assert result_tf.dtype == input_tensor.dtype, (
        f"Dtype mismatch: expected {input_tensor.dtype}, got {result_tf.dtype}"
    )
    
    # Empty tensor and scalar correctness
    np.testing.assert_array_equal(result_np, expected_np,
        f"Boundary condition handling incorrect for pattern: {expected_pattern}")
    
    # Additional validation for empty tensors
    if input_shape == [0]:
        # Empty tensor should remain empty
        assert result_np.size == 0, "Empty tensor should remain empty after roll"
        assert result_np.shape == (0,), "Empty tensor shape should be preserved"
    
    # Additional validation for scalars
    if input_shape == []:
        # Scalar tensor should remain unchanged
        np.testing.assert_array_equal(result_np, input_np,
            "Scalar tensor should remain unchanged after roll")
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
@pytest.mark.parametrize(
    "input_shape,input_dtype,shift,axis,expected_pattern",
    [
        # Base case from test plan
        ([3], tf.float32, "invalid", 0, "非int类型shift引发异常"),
        # Parameter extension
        ([3], tf.float32, [1, 2], [0], "shift/axis形状不匹配引发异常"),
    ]
)
def test_type_validation_error_handling(input_shape, input_dtype, shift, axis, expected_pattern):
    """Test error handling for invalid input types and shapes."""
    # Create test tensor
    input_tensor = create_test_tensor(input_shape, input_dtype)
    
    # Mock the underlying C++ implementation if needed
    # For now, we'll test with actual TensorFlow implementation
    
    # Perform TensorFlow roll operation and expect exception
    from tensorflow.python.ops import manip_ops
    
    if shift == "invalid":
        # Test with invalid shift type (string instead of int)
        with pytest.raises((TypeError, ValueError)) as exc_info:
            manip_ops.roll(input_tensor, shift=shift, axis=axis)
        
        # Weak assertion: exception was raised
        assert exc_info.value is not None, "Expected exception for invalid shift type"
        
        # Check error message contains relevant information
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["type", "int", "shift", "invalid"]), (
            f"Error message should mention type/int/shift, got: {error_msg}"
        )
    
    elif isinstance(shift, list) and isinstance(axis, list) and len(shift) != len(axis):
        # Test with shape mismatch between shift and axis
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)) as exc_info:
            manip_ops.roll(input_tensor, shift=shift, axis=axis)
        
        # Weak assertion: exception was raised
        assert exc_info.value is not None, "Expected exception for shape mismatch"
        
        # Check error message contains relevant information
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["shape", "match", "length", "size"]), (
            f"Error message should mention shape/length mismatch, got: {error_msg}"
        )
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:FOOTER START ====
# Additional helper functions and cleanup
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====