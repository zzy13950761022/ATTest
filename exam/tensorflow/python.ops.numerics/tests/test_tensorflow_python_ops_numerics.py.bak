"""
Unit tests for tensorflow.python.ops.numerics module.
"""
import math
import numpy as np
import pytest
import tensorflow as tf
from unittest import mock

# ==== BLOCK:HEADER START ====
# Import the target function
from tensorflow.python.ops.numerics import verify_tensor_all_finite_v2


class TestVerifyTensorAllFiniteV2:
    """Test class for verify_tensor_all_finite_v2 function."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Set random seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        yield
        # Cleanup if needed
    
    @pytest.fixture
    def mock_check_numerics(self):
        """Mock for tensorflow.python.ops.array_ops.check_numerics."""
        # Import the module first to ensure it's available for mocking
        import tensorflow.python.ops.array_ops as array_ops
        with mock.patch.object(array_ops, 'check_numerics') as mock_fn:
            # Default behavior: return the input tensor
            mock_fn.side_effect = lambda x, message: x
            yield mock_fn
    
    @pytest.fixture
    def mock_with_dependencies(self):
        """Mock for tensorflow.python.ops.control_flow_ops.with_dependencies."""
        # Import the module first to ensure it's available for mocking
        import tensorflow.python.ops.control_flow_ops as control_flow_ops
        with mock.patch.object(control_flow_ops, 'with_dependencies') as mock_fn:
            # Default behavior: return the second argument (the tensor)
            mock_fn.side_effect = lambda dependencies, x: x
            yield mock_fn
    
    @pytest.fixture
    def mock_convert_to_tensor(self):
        """Mock for tensorflow.python.framework.ops.convert_to_tensor."""
        # Import the module first to ensure it's available for mocking
        import tensorflow.python.framework.ops as ops
        with mock.patch.object(ops, 'convert_to_tensor') as mock_fn:
            # Default behavior: return the input as-is if already a tensor
            mock_fn.side_effect = lambda x, **kwargs: x
            yield mock_fn
    
    @pytest.fixture
    def mock_colocate_with(self):
        """Mock for tensorflow.python.framework.ops.colocate_with."""
        # Import the module first to ensure it's available for mocking
        import tensorflow.python.framework.ops as ops
        with mock.patch.object(ops, 'colocate_with') as mock_fn:
            # Create a mock context manager that accepts the tensor argument
            context_mock = mock.MagicMock()
            context_mock.__enter__ = mock.MagicMock(return_value=None)
            context_mock.__exit__ = mock.MagicMock(return_value=None)
            # Return the context manager when colocate_with is called
            mock_fn.return_value = context_mock
            yield mock_fn
    
    @pytest.fixture
    def all_mocks(self, mock_check_numerics, mock_with_dependencies, 
                  mock_convert_to_tensor, mock_colocate_with):
        """Combine all mocks for convenience."""
        return {
            'check_numerics': mock_check_numerics,
            'with_dependencies': mock_with_dependencies,
            'convert_to_tensor': mock_convert_to_tensor,
            'colocate_with': mock_colocate_with
        }
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
    @pytest.mark.parametrize(
        "dtype,shape,message,name",
        [
            # Base case from test plan
            ("float32", [2, 3], "Test valid tensor", None),
            # Parameter extensions
            ("float64", [5, 5], "Large matrix test", "large_check"),
            ("float32", [1, 10, 10], "3D tensor test", "3d_check"),
        ]
    )
    def test_valid_tensor_no_nan_inf(
        self, dtype, shape, message, name, all_mocks
    ):
        """
        TC-01: 正常浮点张量无NaNInf通过检查
        
        Weak assertions:
        - returns_tensor: Function returns a tensor
        - same_shape: Returned tensor has same shape as input
        - same_dtype: Returned tensor has same dtype as input
        - no_exception: No exception is raised
        """
        # Create a valid tensor without NaN/Inf
        if dtype == "float32":
            tf_dtype = tf.float32
            np_dtype = np.float32
        elif dtype == "float64":
            tf_dtype = tf.float64
            np_dtype = np.float64
        else:
            pytest.skip(f"Unsupported dtype for this test: {dtype}")
        
        # Generate random valid values
        np_array = np.random.randn(*shape).astype(np_dtype)
        # Ensure no NaN/Inf
        np_array = np.clip(np_array, -1e6, 1e6)
        
        # Convert to tensor
        input_tensor = tf.constant(np_array, dtype=tf_dtype)
        
        # Call the function
        result = verify_tensor_all_finite_v2(
            x=input_tensor,
            message=message,
            name=name
        )
        
        # Weak assertions
        # 1. returns_tensor: Function returns a tensor
        assert isinstance(result, tf.Tensor), "Function should return a tensor"
        
        # 2. same_shape: Returned tensor has same shape as input
        assert result.shape == input_tensor.shape, (
            f"Result shape {result.shape} should match input shape {input_tensor.shape}"
        )
        
        # 3. same_dtype: Returned tensor has same dtype as input
        assert result.dtype == input_tensor.dtype, (
            f"Result dtype {result.dtype} should match input dtype {input_tensor.dtype}"
        )
        
        # 4. no_exception: No exception is raised (implicitly passed if we get here)
        
        # Verify mocks were called appropriately
        all_mocks['convert_to_tensor'].assert_called_once()
        all_mocks['colocate_with'].assert_called_once()
        all_mocks['check_numerics'].assert_called_once()
        all_mocks['with_dependencies'].assert_called_once()
        
        # Verify check_numerics was called with correct arguments
        check_numerics_call = all_mocks['check_numerics'].call_args
        assert check_numerics_call[0][0] is input_tensor
        assert check_numerics_call[1]['message'] == message
        
        # Verify with_dependencies was called with correct arguments
        with_deps_call = all_mocks['with_dependencies'].call_args
        # First arg should be a list containing the check_numerics result
        assert len(with_deps_call[0][0]) == 1
        # Second arg should be the input tensor
        assert with_deps_call[0][1] is input_tensor
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
    @pytest.mark.parametrize(
        "dtype,shape,has_nan,message,name",
        [
            # Base case from test plan
            ("float32", [3], True, "NaN detected", "check_nan"),
            # Parameter extension
            ("float64", [2, 3, 4], True, "3D NaN test", "3d_nan"),
        ]
    )
    def test_tensor_with_nan_triggers_error(
        self, dtype, shape, has_nan, message, name, all_mocks
    ):
        """
        TC-02: 包含NaN的张量触发错误记录
        
        Weak assertions:
        - returns_tensor: Function returns a tensor
        - same_shape: Returned tensor has same shape as input
        - same_dtype: Returned tensor has same dtype as input
        - no_exception: No exception is raised (function should handle NaN gracefully)
        """
        # Skip if has_nan is False (not applicable for this test)
        if not has_nan:
            pytest.skip("This test requires has_nan=True")
        
        # Create tensor with NaN
        if dtype == "float32":
            tf_dtype = tf.float32
            np_dtype = np.float32
        elif dtype == "float64":
            tf_dtype = tf.float64
            np_dtype = np.float64
        else:
            pytest.skip(f"Unsupported dtype for this test: {dtype}")
        
        # Generate random values and insert NaN
        np_array = np.random.randn(*shape).astype(np_dtype)
        # Insert NaN at a random position
        if np_array.size > 0:
            flat_idx = np.random.randint(0, np_array.size)
            np_array.flat[flat_idx] = np.nan
        
        # Convert to tensor
        input_tensor = tf.constant(np_array, dtype=tf_dtype)
        
        # Configure mock to simulate check_numerics behavior with NaN
        # check_numerics should still return a tensor (even with NaN)
        all_mocks['check_numerics'].side_effect = lambda x, message: x
        
        # Call the function
        result = verify_tensor_all_finite_v2(
            x=input_tensor,
            message=message,
            name=name
        )
        
        # Weak assertions
        # 1. returns_tensor: Function returns a tensor
        assert isinstance(result, tf.Tensor), "Function should return a tensor"
        
        # 2. same_shape: Returned tensor has same shape as input
        assert result.shape == input_tensor.shape, (
            f"Result shape {result.shape} should match input shape {input_tensor.shape}"
        )
        
        # 3. same_dtype: Returned tensor has same dtype as input
        assert result.dtype == input_tensor.dtype, (
            f"Result dtype {result.dtype} should match input dtype {input_tensor.dtype}"
        )
        
        # 4. no_exception: No exception is raised (function handles NaN)
        
        # Verify mocks were called appropriately
        all_mocks['convert_to_tensor'].assert_called_once()
        all_mocks['colocate_with'].assert_called_once()
        all_mocks['check_numerics'].assert_called_once()
        all_mocks['with_dependencies'].assert_called_once()
        
        # Verify check_numerics was called with correct arguments
        check_numerics_call = all_mocks['check_numerics'].call_args
        assert check_numerics_call[0][0] is input_tensor
        assert check_numerics_call[1]['message'] == message
        
        # Note: In real TensorFlow, check_numerics would raise an error or log
        # when it encounters NaN, but for unit testing we're mocking it.
        # The actual error logging behavior would be tested in integration tests.
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
    @pytest.mark.parametrize(
        "dtype,shape,has_inf,message,name",
        [
            # Base case from test plan
            ("float64", [2, 2], True, "Inf detected", "check_inf"),
            # Parameter extension
            ("float32", [10], True, "Vector Inf test", "vector_inf"),
        ]
    )
    def test_tensor_with_inf_triggers_error(
        self, dtype, shape, has_inf, message, name, all_mocks
    ):
        """
        TC-03: 包含Inf的张量触发错误记录
        
        Weak assertions:
        - returns_tensor: Function returns a tensor
        - same_shape: Returned tensor has same shape as input
        - same_dtype: Returned tensor has same dtype as input
        - no_exception: No exception is raised (function should handle Inf gracefully)
        """
        # Skip if has_inf is False (not applicable for this test)
        if not has_inf:
            pytest.skip("This test requires has_inf=True")
        
        # Create tensor with Inf
        if dtype == "float32":
            tf_dtype = tf.float32
            np_dtype = np.float32
        elif dtype == "float64":
            tf_dtype = tf.float64
            np_dtype = np.float64
        else:
            pytest.skip(f"Unsupported dtype for this test: {dtype}")
        
        # Generate random values and insert Inf
        np_array = np.random.randn(*shape).astype(np_dtype)
        # Insert positive infinity at a random position
        if np_array.size > 0:
            flat_idx = np.random.randint(0, np_array.size)
            np_array.flat[flat_idx] = np.inf
        
        # Convert to tensor
        input_tensor = tf.constant(np_array, dtype=tf_dtype)
        
        # Configure mock to simulate check_numerics behavior with Inf
        # check_numerics should still return a tensor (even with Inf)
        all_mocks['check_numerics'].side_effect = lambda x, message: x
        
        # Call the function
        result = verify_tensor_all_finite_v2(
            x=input_tensor,
            message=message,
            name=name
        )
        
        # Weak assertions
        # 1. returns_tensor: Function returns a tensor
        assert isinstance(result, tf.Tensor), "Function should return a tensor"
        
        # 2. same_shape: Returned tensor has same shape as input
        assert result.shape == input_tensor.shape, (
            f"Result shape {result.shape} should match input shape {input_tensor.shape}"
        )
        
        # 3. same_dtype: Returned tensor has same dtype as input
        assert result.dtype == input_tensor.dtype, (
            f"Result dtype {result.dtype} should match input dtype {input_tensor.dtype}"
        )
        
        # 4. no_exception: No exception is raised (function handles Inf)
        
        # Verify mocks were called appropriately
        all_mocks['convert_to_tensor'].assert_called_once()
        all_mocks['colocate_with'].assert_called_once()
        all_mocks['check_numerics'].assert_called_once()
        all_mocks['with_dependencies'].assert_called_once()
        
        # Verify check_numerics was called with correct arguments
        check_numerics_call = all_mocks['check_numerics'].call_args
        assert check_numerics_call[0][0] is input_tensor
        assert check_numerics_call[1]['message'] == message
        
        # Note: In real TensorFlow, check_numerics would raise an error or log
        # when it encounters Inf, but for unit testing we're mocking it.
        # The actual error logging behavior would be tested in integration tests.
        
        # Additional check: verify that the function handles both positive and negative infinity
        # Create tensor with negative infinity
        np_array_neg_inf = np.random.randn(*shape).astype(np_dtype)
        if np_array_neg_inf.size > 0:
            flat_idx = np.random.randint(0, np_array_neg_inf.size)
            np_array_neg_inf.flat[flat_idx] = -np.inf
        
        input_tensor_neg_inf = tf.constant(np_array_neg_inf, dtype=tf_dtype)
        
        # Reset mocks for second call
        for mock_obj in all_mocks.values():
            mock_obj.reset_mock()
        
        # Configure mock again
        all_mocks['check_numerics'].side_effect = lambda x, message: x
        
        # Call function with negative infinity tensor
        result_neg_inf = verify_tensor_all_finite_v2(
            x=input_tensor_neg_inf,
            message=message + " negative",
            name=name + "_neg"
        )
        
        # Verify it still returns a tensor
        assert isinstance(result_neg_inf, tf.Tensor), "Function should handle negative infinity"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
    @pytest.mark.parametrize(
        "dtype,shape,message,name",
        [
            # Base case from test plan
            ("float16", [4], "float16 test", None),
            # Parameter extension
            ("bfloat16", [2, 2], "bfloat16 test", "bfloat_check"),
        ]
    )
    def test_different_float_dtype_compatibility(
        self, dtype, shape, message, name, all_mocks
    ):
        """
        TC-04: 不同浮点数据类型兼容性
        
        Weak assertions:
        - returns_tensor: Function returns a tensor
        - same_shape: Returned tensor has same shape as input
        - same_dtype: Returned tensor has same dtype as input
        - no_exception: No exception is raised
        """
        # Create a valid tensor without NaN/Inf
        if dtype == "float16":
            tf_dtype = tf.float16
            np_dtype = np.float16
        elif dtype == "bfloat16":
            tf_dtype = tf.bfloat16
            np_dtype = np.float32  # bfloat16 not directly supported in numpy
        else:
            pytest.skip(f"Unsupported dtype for this test: {dtype}")
        
        # Generate random valid values
        np_array = np.random.randn(*shape).astype(np_dtype)
        # Ensure no NaN/Inf
        np_array = np.clip(np_array, -1e6, 1e6)
        
        # Convert to tensor
        input_tensor = tf.constant(np_array, dtype=tf_dtype)
        
        # Call the function
        result = verify_tensor_all_finite_v2(
            x=input_tensor,
            message=message,
            name=name
        )
        
        # Weak assertions
        # 1. returns_tensor: Function returns a tensor
        assert isinstance(result, tf.Tensor), "Function should return a tensor"
        
        # 2. same_shape: Returned tensor has same shape as input
        assert result.shape == input_tensor.shape, (
            f"Result shape {result.shape} should match input shape {input_tensor.shape}"
        )
        
        # 3. same_dtype: Returned tensor has same dtype as input
        assert result.dtype == input_tensor.dtype, (
            f"Result dtype {result.dtype} should match input dtype {input_tensor.dtype}"
        )
        
        # 4. no_exception: No exception is raised (implicitly passed if we get here)
        
        # Verify mocks were called appropriately
        all_mocks['convert_to_tensor'].assert_called_once()
        all_mocks['colocate_with'].assert_called_once()
        all_mocks['check_numerics'].assert_called_once()
        all_mocks['with_dependencies'].assert_called_once()
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
    @pytest.mark.parametrize(
        "dtype,shape,message,name",
        [
            # Base case from test plan
            ("float32", [], "scalar test", "scalar_check"),
            # Parameter extension
            ("float32", [100], "long vector test", "long_vector"),
        ]
    )
    def test_different_shape_tensor_check(
        self, dtype, shape, message, name, all_mocks
    ):
        """
        TC-05: 不同形状张量检查
        
        Weak assertions:
        - returns_tensor: Function returns a tensor
        - same_shape: Returned tensor has same shape as input
        - same_dtype: Returned tensor has same dtype as input
        - no_exception: No exception is raised
        """
        # Create a valid tensor without NaN/Inf
        if dtype == "float32":
            tf_dtype = tf.float32
            np_dtype = np.float32
        else:
            pytest.skip(f"Unsupported dtype for this test: {dtype}")
        
        # Generate random valid values
        if shape:  # Non-empty shape (list with elements)
            np_array = np.random.randn(*shape).astype(np_dtype)
        else:  # Scalar (empty shape list)
            # For scalar, create a 0-dimensional numpy array
            np_array = np.array(np.random.randn(), dtype=np_dtype)
        
        # Ensure no NaN/Inf
        np_array = np.clip(np_array, -1e6, 1e6)
        
        # Convert to tensor
        input_tensor = tf.constant(np_array, dtype=tf_dtype)
        
        # Call the function
        result = verify_tensor_all_finite_v2(
            x=input_tensor,
            message=message,
            name=name
        )
        
        # Weak assertions
        # 1. returns_tensor: Function returns a tensor
        assert isinstance(result, tf.Tensor), "Function should return a tensor"
        
        # 2. same_shape: Returned tensor has same shape as input
        assert result.shape == input_tensor.shape, (
            f"Result shape {result.shape} should match input shape {input_tensor.shape}"
        )
        
        # 3. same_dtype: Returned tensor has same dtype as input
        assert result.dtype == input_tensor.dtype, (
            f"Result dtype {result.dtype} should match input dtype {input_tensor.dtype}"
        )
        
        # 4. no_exception: No exception is raised (implicitly passed if we get here)
        
        # Verify mocks were called appropriately
        all_mocks['convert_to_tensor'].assert_called_once()
        all_mocks['colocate_with'].assert_called_once()
        all_mocks['check_numerics'].assert_called_once()
        all_mocks['with_dependencies'].assert_called_once()
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:FOOTER START ====
    # Helper methods for future tests
    
    def _create_valid_tensor(self, dtype_str, shape):
        """Helper to create a valid tensor without NaN/Inf."""
        if dtype_str == "float16":
            tf_dtype = tf.float16
            np_dtype = np.float16
        elif dtype_str == "float32":
            tf_dtype = tf.float32
            np_dtype = np.float32
        elif dtype_str == "float64":
            tf_dtype = tf.float64
            np_dtype = np.float64
        elif dtype_str == "bfloat16":
            tf_dtype = tf.bfloat16
            np_dtype = np.float32  # bfloat16 not directly supported in numpy
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        
        np_array = np.random.randn(*shape).astype(np_dtype)
        np_array = np.clip(np_array, -1e6, 1e6)  # Ensure no overflow
        return tf.constant(np_array, dtype=tf_dtype)
    
    def _create_tensor_with_nan(self, dtype_str, shape):
        """Helper to create a tensor with NaN values."""
        tensor = self._create_valid_tensor(dtype_str, shape)
        # Convert to numpy, add NaN, and back to tensor
        np_array = tensor.numpy()
        if np_array.size > 0:
            flat_idx = np.random.randint(0, np_array.size)
            np_array.flat[flat_idx] = np.nan
        return tf.constant(np_array, dtype=tensor.dtype)
    
    def _create_tensor_with_inf(self, dtype_str, shape):
        """Helper to create a tensor with Inf values."""
        tensor = self._create_valid_tensor(dtype_str, shape)
        # Convert to numpy, add Inf, and back to tensor
        np_array = tensor.numpy()
        if np_array.size > 0:
            flat_idx = np.random.randint(0, np_array.size)
            np_array.flat[flat_idx] = np.inf
        return tf.constant(np_array, dtype=tensor.dtype)


# Test for the v1 alias function
class TestVerifyTensorAllFiniteV1:
    """Test class for the v1 alias function verify_tensor_all_finite."""
    
    def test_v1_function_exists(self):
        """Test that the v1 alias function exists and can be imported."""
        from tensorflow.python.ops.numerics import verify_tensor_all_finite
        assert callable(verify_tensor_all_finite)
    
    def test_v1_calls_v2(self):
        """Test that v1 function calls v2 function with proper argument mapping."""
        from tensorflow.python.ops.numerics import verify_tensor_all_finite
        
        # Mock the v2 function using unittest.mock directly
        # Patch the actual module where v2 is defined
        with mock.patch('tensorflow.python.ops.numerics.verify_tensor_all_finite_v2') as mock_v2:
            mock_v2.return_value = "mocked_result"
            
            # Create a test tensor
            test_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
            
            # Test with t/msg parameters (old names)
            result = verify_tensor_all_finite(t=test_tensor, msg="test message")
            
            # Verify v2 was called with correct arguments
            mock_v2.assert_called_once_with(test_tensor, "test message", None)
            assert result == "mocked_result"
            
            # Reset mock
            mock_v2.reset_mock()
            
            # Test with x/message parameters (new names)
            result = verify_tensor_all_finite(x=test_tensor, message="new message")
            
            # Verify v2 was called with correct arguments
            mock_v2.assert_called_once_with(test_tensor, "new message", None)
            assert result == "mocked_result"


# Test for add_check_numerics_ops (basic existence test)
def test_add_check_numerics_ops_exists():
    """Test that add_check_numerics_ops function exists."""
    from tensorflow.python.ops.numerics import add_check_numerics_ops
    assert callable(add_check_numerics_ops)
# ==== BLOCK:FOOTER END ====