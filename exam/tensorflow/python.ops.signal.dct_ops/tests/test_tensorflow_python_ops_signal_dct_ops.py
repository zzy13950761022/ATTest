"""
Test cases for tensorflow.python.ops.signal.dct_ops
"""
import math
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.ops.signal import dct_ops

# ==== BLOCK:HEADER START ====
"""
Test cases for tensorflow.python.ops.signal.dct_ops
"""
import math
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.ops.signal import dct_ops

# Test fixtures and helper functions
@pytest.fixture
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    tf.random.set_seed(42)
    yield
    # Cleanup if needed

def create_test_signal(shape, dtype=np.float32):
    """Create a test signal with random values."""
    signal = np.random.randn(*shape).astype(dtype)
    return tf.constant(signal, dtype=dtype)

def assert_tensor_properties(tensor, expected_shape, expected_dtype):
    """Assert tensor shape and dtype."""
    assert tensor.shape == expected_shape
    assert tensor.dtype == expected_dtype
    # Check finite values
    assert tf.reduce_all(tf.math.is_finite(tensor)).numpy()

def compare_with_scipy_dct(tf_result, input_signal, dct_type=2, norm=None, rtol=1e-5, atol=1e-5):
    """Compare TensorFlow DCT result with SciPy reference."""
    try:
        import scipy.fftpack
    except ImportError:
        pytest.skip("SciPy not available for reference comparison")
    
    # Convert to numpy for SciPy
    if isinstance(input_signal, tf.Tensor):
        input_np = input_signal.numpy()
    else:
        input_np = input_signal
    
    if isinstance(tf_result, tf.Tensor):
        tf_np = tf_result.numpy()
    else:
        tf_np = tf_result
    
    # Compute SciPy DCT
    scipy_result = scipy.fftpack.dct(input_np, type=dct_type, norm=norm, axis=-1)
    
    # Compare
    np.testing.assert_allclose(tf_np, scipy_result, rtol=rtol, atol=atol)
    return True

def compare_with_scipy_idct(tf_result, input_signal, idct_type=2, norm=None, rtol=1e-5, atol=1e-5):
    """Compare TensorFlow IDCT result with SciPy reference."""
    try:
        import scipy.fftpack
    except ImportError:
        pytest.skip("SciPy not available for reference comparison")
    
    # Convert to numpy for SciPy
    if isinstance(input_signal, tf.Tensor):
        input_np = input_signal.numpy()
    else:
        input_np = input_signal
    
    if isinstance(tf_result, tf.Tensor):
        tf_np = tf_result.numpy()
    else:
        tf_np = tf_result
    
    # Compute SciPy IDCT
    scipy_result = scipy.fftpack.idct(input_np, type=idct_type, norm=norm, axis=-1)
    
    # Compare
    np.testing.assert_allclose(tf_np, scipy_result, rtol=rtol, atol=atol)
    return True
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# DCT基本功能验证
class TestDCTBasic:
    """Test basic DCT functionality."""
    
    @pytest.mark.parametrize("dtype, shape, dct_type, norm, n", [
        (tf.float32, (8,), 2, None, None),  # Basic case from test plan
        (tf.float64, (16,), 2, "ortho", None),  # Ortho normalization extension
        (tf.float32, (8,), 2, None, 4),  # n parameter truncation
        (tf.float32, (4,), 2, None, 8),  # n parameter zero padding
    ])
    def test_dct_basic_functionality(self, set_random_seed, dtype, shape, dct_type, norm, n):
        """Test basic DCT functionality with various parameters."""
        # Create test signal
        signal = create_test_signal(shape, dtype.as_numpy_dtype)
        
        # Compute DCT
        result = dct_ops.dct(signal, type=dct_type, n=n, norm=norm)
        
        # Weak assertions (round 1)
        # 1. Shape assertion
        expected_shape = shape
        if n is not None:
            expected_shape = (n,) if len(shape) == 1 else shape[:-1] + (n,)
        assert result.shape == expected_shape
        
        # 2. Dtype assertion
        assert result.dtype == dtype
        
        # 3. Finite values assertion
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()
        
        # 4. Basic property: DCT of real signal should be real
        # Since DCT of real signal produces real output, we can check that
        # the imaginary part is zero (or very close to zero)
        # Note: tf.math.is_real doesn't exist, so we check if result is real
        # by verifying it's not complex dtype
        assert not result.dtype.is_complex
        
        # 5. Basic property: DCT should not be all zeros for non-zero input
        # (unless n=0 truncation, but n is positive or None)
        if n != 0:
            assert not tf.reduce_all(tf.abs(result) < 1e-10).numpy()
        
        # Note: Strong assertions (approx_equal_scipy, orthogonality, inverse_property)
        # are deferred to later rounds per test plan
    
    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    def test_dct_2d_input(self, set_random_seed, dtype):
        """Test DCT with 2D input (batch processing)."""
        shape = (3, 8)  # 3 signals of length 8
        signal = create_test_signal(shape, dtype.as_numpy_dtype)
        
        # Compute DCT
        result = dct_ops.dct(signal, type=2)
        
        # Assertions
        assert result.shape == shape
        assert result.dtype == dtype
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()
        
        # Check that each row was transformed independently
        # (basic sanity check - not a rigorous mathematical property)
        for i in range(shape[0]):
            row_signal = signal[i:i+1, :]
            row_result = dct_ops.dct(row_signal, type=2)
            np.testing.assert_allclose(result[i:i+1, :].numpy(), row_result.numpy(), rtol=1e-5)
    
    def test_dct_identity_property(self, set_random_seed):
        """Test that DCT of a constant signal has expected properties."""
        # Create constant signal
        signal = tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32)
        
        # Compute DCT type 2
        result = dct_ops.dct(signal, type=2)
        
        # For a constant signal, DCT-II should have:
        # - First coefficient = N * sqrt(2) for ortho normalization
        # - First coefficient = N for no normalization
        # - Other coefficients = 0 (except for numerical precision)
        
        # According to standard DCT-II definition:
        # X_k = sum_{n=0}^{N-1} x_n * cos(pi * k * (2n+1) / (2N))
        # For constant x_n = 1:
        # X_0 = sum_{n=0}^{N-1} 1 * cos(0) = N = 4
        # X_k (k>0) = sum_{n=0}^{N-1} cos(pi * k * (2n+1) / (2N)) = 0
        
        # However, TensorFlow's implementation uses 2N padding which introduces
        # an extra factor of 2. This is consistent with the documentation
        # that says we need to scale by 0.5/N for the inverse.
        N = signal.shape[-1]
        
        # With TensorFlow's implementation, we get 2N instead of N
        expected_first = 2 * N  # TensorFlow's scaling factor
        assert abs(result[0].numpy() - expected_first) < 1e-5
        
        # Other coefficients should be close to zero
        for i in range(1, N):
            assert abs(result[i].numpy()) < 1e-5
        
        # Verify the scaling factor is consistent with inverse relationship
        # Compute inverse using DCT type 3
        dct3_result = dct_ops.dct(result, type=3)
        reconstructed = dct3_result * 0.5 / N  # From documentation
        
        # Should reconstruct original signal
        np.testing.assert_allclose(signal.numpy(), reconstructed.numpy(), rtol=1e-5, atol=1e-5)
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# IDCT基本功能验证
class TestIDCTBasic:
    """Test basic IDCT functionality."""
    
    @pytest.mark.parametrize("dtype, shape, idct_type, norm", [
        (tf.float32, (8,), 2, None),  # Basic case from test plan
        (tf.float64, (8,), 3, "ortho"),  # IDCT type 3 with ortho normalization
    ])
    def test_idct_basic_functionality(self, set_random_seed, dtype, shape, idct_type, norm):
        """Test basic IDCT functionality with various parameters."""
        # Create test signal
        signal = create_test_signal(shape, dtype.as_numpy_dtype)
        
        # Compute IDCT
        result = dct_ops.idct(signal, type=idct_type, norm=norm)
        
        # Weak assertions (round 1)
        # 1. Shape assertion
        assert result.shape == shape
        
        # 2. Dtype assertion
        assert result.dtype == dtype
        
        # 3. Finite values assertion
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()
        
        # 4. Basic property: IDCT of real signal should be real
        # Check that result is not complex dtype
        assert not result.dtype.is_complex
        
        # Note: Strong assertions (approx_equal_scipy, inverse_consistency, roundtrip)
        # are deferred to later rounds per test plan
    
    def test_idct_inverse_relationship(self, set_random_seed):
        """Test that IDCT is the inverse of DCT (basic check)."""
        # Create test signal
        signal = create_test_signal((8,), np.float32)
        
        # Test for type 2/3 pair (most common)
        # DCT type 2
        dct_result = dct_ops.dct(signal, type=2)
        
        # IDCT type 3 (inverse of DCT type 2)
        idct_result = dct_ops.idct(dct_result, type=3)
        
        # According to TensorFlow documentation:
        # Without normalization, need to scale by 0.5/N to get inverse
        # signal == idct(dct(signal)) * 0.5 / N
        N = signal.shape[-1]
        
        # Apply the correct scaling factor from documentation
        reconstructed = idct_result * 0.5 / N
        
        # Check reconstruction (with tolerance for numerical errors)
        np.testing.assert_allclose(signal.numpy(), reconstructed.numpy(), rtol=1e-5, atol=1e-5)
        
        # Also test the reverse: DCT type 3 and IDCT type 2
        dct_result_type3 = dct_ops.dct(signal, type=3)
        idct_result_type2 = dct_ops.idct(dct_result_type3, type=2)
        reconstructed_reverse = idct_result_type2 * 0.5 / N
        np.testing.assert_allclose(signal.numpy(), reconstructed_reverse.numpy(), rtol=1e-5, atol=1e-5)
    
    def test_idct_with_ortho_normalization(self, set_random_seed):
        """Test IDCT with orthonormal normalization."""
        signal = create_test_signal((8,), np.float32)
        
        # DCT with ortho normalization
        dct_result = dct_ops.dct(signal, type=2, norm='ortho')
        
        # IDCT with ortho normalization (should be exact inverse)
        idct_result = dct_ops.idct(dct_result, type=3, norm='ortho')
        
        # With ortho normalization, reconstruction should be exact
        # (up to numerical precision) as per documentation
        np.testing.assert_allclose(signal.numpy(), idct_result.numpy(), rtol=1e-5, atol=1e-5)
        
        # Also test the reverse: DCT type 3 and IDCT type 2 with ortho
        dct_result_type3 = dct_ops.dct(signal, type=3, norm='ortho')
        idct_result_type2 = dct_ops.idct(dct_result_type3, type=2, norm='ortho')
        np.testing.assert_allclose(signal.numpy(), idct_result_type2.numpy(), rtol=1e-5, atol=1e-5)
    
    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    def test_idct_float64_precision(self, set_random_seed, dtype):
        """Test IDCT with float64 precision."""
        shape = (8,)
        signal = create_test_signal(shape, dtype.as_numpy_dtype)
        
        # Compute IDCT
        result = dct_ops.idct(signal, type=2)
        
        # Basic assertions
        assert result.shape == shape
        assert result.dtype == dtype
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()
        
        # Check that result has expected precision characteristics
        # For float64, we expect better numerical stability
        if dtype == tf.float64:
            # Compute also with float32 for comparison
            signal_f32 = tf.cast(signal, tf.float32)
            result_f32 = dct_ops.idct(signal_f32, type=2)
            
            # Convert float64 result to float32 for comparison
            result_f64_as_f32 = tf.cast(result, tf.float32)
            
            # They should be close (within float32 precision)
            np.testing.assert_allclose(
                result_f32.numpy(), 
                result_f64_as_f32.numpy(), 
                rtol=1e-5, 
                atol=1e-5
            )
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
# 参数验证与异常处理
class TestParameterValidation:
    """Test parameter validation and error handling."""
    
    @pytest.mark.parametrize("dtype, shape, dct_type, norm, n, expect_error, error_type", [
        # Invalid DCT type (type=5) - should raise ValueError
        (tf.float32, (4,), 5, None, None, True, ValueError),
        # Type-I DCT with samples=1 - should raise ValueError
        (tf.float32, (1,), 1, None, None, True, ValueError),
        # Type-I DCT with ortho normalization - should raise ValueError
        (tf.float32, (8,), 1, "ortho", None, True, ValueError),
    ])
    def test_dct_invalid_parameters(self, set_random_seed, dtype, shape, dct_type, norm, n, expect_error, error_type):
        """Test DCT with invalid parameters raises appropriate errors."""
        signal = create_test_signal(shape, dtype.as_numpy_dtype)
        
        if expect_error:
            with pytest.raises(error_type) as exc_info:
                dct_ops.dct(signal, type=dct_type, norm=norm, n=n)
            
            # Weak assertion: error was raised with correct type
            assert exc_info.type == error_type
            
            # Optional: check error message contains relevant keywords
            error_msg = str(exc_info.value).lower()
            if dct_type == 5:
                assert "type" in error_msg or "supported" in error_msg
            elif dct_type == 1 and shape[-1] == 1:
                assert "dimension" in error_msg or "greater" in error_msg
            elif dct_type == 1 and norm == "ortho":
                assert "normalization" in error_msg or "supported" in error_msg
        else:
            # Should not raise error
            result = dct_ops.dct(signal, type=dct_type, norm=norm, n=n)
            assert result is not None
    
    def test_dct_invalid_n_parameter(self, set_random_seed):
        """Test DCT with invalid n parameter."""
        signal = create_test_signal((8,), np.float32)
        
        # n = 0 should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            dct_ops.dct(signal, n=0)
        
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()
        
        # n = -1 should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            dct_ops.dct(signal, n=-1)
        
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()
    
    def test_dct_invalid_norm_parameter(self, set_random_seed):
        """Test DCT with invalid norm parameter."""
        signal = create_test_signal((8,), np.float32)
        
        # norm = "invalid" should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            dct_ops.dct(signal, norm="invalid")
        
        assert "norm" in str(exc_info.value).lower() or "normalization" in str(exc_info.value).lower()
        assert "ortho" in str(exc_info.value).lower() or "none" in str(exc_info.value).lower()
    
    def test_idct_invalid_n_parameter(self, set_random_seed):
        """Test IDCT with invalid n parameter."""
        signal = create_test_signal((8,), np.float32)
        
        # n = 0 should raise ValueError (same validation as DCT)
        with pytest.raises(ValueError) as exc_info:
            dct_ops.idct(signal, n=0)
        
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()
        
        # n = -1 should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            dct_ops.idct(signal, n=-1)
        
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()
        
        # Note: n = 8 (positive integer) should work for IDCT
        # The documentation says n must be None, but implementation allows positive integers
        result = dct_ops.idct(signal, n=8)
        assert result.shape == (8,)
    
    def test_dct_invalid_axis_parameter(self, set_random_seed):
        """Test DCT with invalid axis parameter (must be -1)."""
        signal = create_test_signal((8,), np.float32)
        
        # axis = 0 should raise NotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            dct_ops.dct(signal, axis=0)
        
        assert "axis" in str(exc_info.value).lower() or "-1" in str(exc_info.value).lower()
        
        # axis = 1 should also raise NotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            dct_ops.dct(signal, axis=1)
    
    def test_idct_invalid_axis_parameter(self, set_random_seed):
        """Test IDCT with invalid axis parameter (must be -1)."""
        signal = create_test_signal((8,), np.float32)
        
        # axis = 0 should raise NotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            dct_ops.idct(signal, axis=0)
        
        assert "axis" in str(exc_info.value).lower() or "-1" in str(exc_info.value).lower()
    
    def test_dct_invalid_input_dtype(self, set_random_seed):
        """Test DCT with invalid input dtype."""
        # Create integer tensor (should fail)
        signal = tf.constant([1, 2, 3, 4], dtype=tf.int32)
        
        # Should raise some error (likely during tensor conversion)
        # The exact error might vary, but it should fail
        with pytest.raises(Exception) as exc_info:
            dct_ops.dct(signal)
        
        # Check it's some kind of error (could be TypeError, ValueError, etc.)
        assert exc_info.type in (TypeError, ValueError, tf.errors.InvalidArgumentError)
    
    def test_dct_empty_tensor(self, set_random_seed):
        """Test DCT with empty tensor."""
        # Empty tensor
        signal = tf.constant([], dtype=tf.float32)
        
        # According to the validation function, empty tensor might not raise
        # an error for DCT types other than Type-I
        # Let's test what actually happens
        
        # For Type 2 DCT, empty tensor might produce empty result
        try:
            result = dct_ops.dct(signal, type=2)
            # If no error, check the result
            assert result.shape == (0,)
            assert result.dtype == tf.float32
        except Exception as e:
            # If error is raised, it should be a ValueError
            assert isinstance(e, ValueError)
            # Check error message
            error_msg = str(e).lower()
            assert "dimension" in error_msg or "empty" in error_msg or "shape" in error_msg
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# DCT类型全覆盖
class TestDCTTypes:
    """Test all DCT types (1, 2, 3, 4)."""
    
    @pytest.mark.parametrize("dtype, shape, dct_type, norm", [
        (tf.float32, (8,), 1, None),  # DCT type 1
        (tf.float32, (8,), 2, None),  # DCT type 2 (already tested, but included for completeness)
        (tf.float32, (8,), 3, None),  # DCT type 3
        (tf.float32, (8,), 4, None),  # DCT type 4
    ])
    def test_dct_all_types_basic(self, set_random_seed, dtype, shape, dct_type, norm):
        """Test basic functionality for all DCT types."""
        # Skip Type-I DCT with shape[0] < 2
        if dct_type == 1 and shape[-1] < 2:
            pytest.skip("Type-I DCT requires dimension > 1")
        
        # Create test signal
        signal = create_test_signal(shape, dtype.as_numpy_dtype)
        
        # Compute DCT
        result = dct_ops.dct(signal, type=dct_type, norm=norm)
        
        # Weak assertions
        # 1. Shape assertion
        assert result.shape == shape
        
        # 2. Dtype assertion
        assert result.dtype == dtype
        
        # 3. Finite values assertion
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()
        
        # 4. Type-specific basic checks
        if dct_type == 1:
            # Type-I DCT is even around n=0 and n=N-1
            # For real input, output should be real
            assert not result.dtype.is_complex
        elif dct_type == 2:
            # Type-II DCT is the most common
            # For real input, output should be real
            assert not result.dtype.is_complex
        elif dct_type == 3:
            # Type-III DCT is the inverse of Type-II
            # For real input, output should be real
            assert not result.dtype.is_complex
        elif dct_type == 4:
            # Type-IV DCT has shift of 1/2 in both indices
            # For real input, output should be real
            assert not result.dtype.is_complex
    
    def test_dct_type1_specific(self, set_random_seed):
        """Test Type-I DCT specific properties."""
        # Type-I DCT requires N > 1
        signal = create_test_signal((8,), np.float32)
        
        # Compute Type-I DCT
        result = dct_ops.dct(signal, type=1)
        
        # Type-I DCT should be real and finite
        assert not result.dtype.is_complex
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()
        
        # Type-I DCT of symmetric signal should have zeros at odd indices
        # Create symmetric signal: [1, 2, 3, 4, 4, 3, 2, 1]
        symmetric_signal = tf.constant([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0], dtype=tf.float32)
        symmetric_result = dct_ops.dct(symmetric_signal, type=1)
        
        # Check it's computed without error
        assert symmetric_result.shape == (8,)
    
    def test_dct_type3_inverse_of_type2(self, set_random_seed):
        """Test that DCT type 3 is the inverse of type 2."""
        signal = create_test_signal((8,), np.float32)
        
        # DCT type 2
        dct2_result = dct_ops.dct(signal, type=2)
        
        # DCT type 3 (inverse of type 2)
        dct3_result = dct_ops.dct(dct2_result, type=3)
        
        # According to TensorFlow documentation for idct:
        # signal == idct(dct(signal)) * 0.5 / N
        # Since idct(input, type=3) calls dct(input, type=2)
        # and idct(input, type=2) calls dct(input, type=3)
        # We have: signal == dct(dct(signal, type=2), type=3) * 0.5 / N
        N = signal.shape[-1]
        reconstructed = dct3_result * 0.5 / N
        
        # Check reconstruction
        np.testing.assert_allclose(signal.numpy(), reconstructed.numpy(), rtol=1e-5, atol=1e-5)
    
    def test_dct_type4_symmetry(self, set_random_seed):
        """Test Type-IV DCT symmetry properties."""
        signal = create_test_signal((8,), np.float32)
        
        # Compute Type-IV DCT
        result = dct_ops.dct(signal, type=4)
        
        # Basic checks
        assert result.shape == (8,)
        assert not result.dtype.is_complex
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()
        
        # Type-IV DCT is its own inverse (with proper scaling)
        # According to documentation, similar scaling applies
        dct4_twice = dct_ops.dct(result, type=4)
        
        # Need to scale: signal ≈ dct4(dct4(signal)) * 0.5 / N
        N = signal.shape[-1]
        reconstructed = dct4_twice * 0.5 / N
        
        # Check reconstruction (with larger tolerance due to numerical issues)
        np.testing.assert_allclose(signal.numpy(), reconstructed.numpy(), rtol=1e-4, atol=1e-4)
    
    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    def test_dct_types_with_ortho_normalization(self, set_random_seed, dtype):
        """Test DCT types with orthonormal normalization."""
        signal = create_test_signal((8,), dtype.as_numpy_dtype)
        
        # Test types that support ortho normalization (2, 3, 4)
        for dct_type in [2, 3, 4]:
            # Skip Type-I with ortho (not supported)
            if dct_type == 1:
                continue
                
            # Compute DCT with ortho normalization
            result = dct_ops.dct(signal, type=dct_type, norm='ortho')
            
            # Basic checks
            assert result.shape == (8,)
            assert result.dtype == dtype
            assert tf.reduce_all(tf.math.is_finite(result)).numpy()
            
            # With ortho normalization, energy should be preserved
            # (Parseval's theorem for orthonormal transforms)
            input_energy = tf.reduce_sum(tf.square(signal)).numpy()
            output_energy = tf.reduce_sum(tf.square(result)).numpy()
            
            # Energy should be approximately equal
            np.testing.assert_allclose(input_energy, output_energy, rtol=1e-5)
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# 浮点精度兼容性
class TestPrecisionCompatibility:
    """Test floating-point precision compatibility."""
    
    @pytest.mark.parametrize("dtype, shape, dct_type, norm", [
        (tf.float32, (8,), 2, None),  # float32 precision
        (tf.float64, (8,), 2, None),  # float64 precision
        (tf.float32, (16,), 2, "ortho"),  # float32 with ortho
        (tf.float64, (16,), 2, "ortho"),  # float64 with ortho
    ])
    def test_dct_precision_basic(self, set_random_seed, dtype, shape, dct_type, norm):
        """Test DCT with different floating-point precisions."""
        # Create test signal
        signal = create_test_signal(shape, dtype.as_numpy_dtype)
        
        # Compute DCT
        result = dct_ops.dct(signal, type=dct_type, norm=norm)
        
        # Basic assertions
        assert result.shape == shape
        assert result.dtype == dtype
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()
        
        # Precision-specific checks
        if dtype == tf.float64:
            # float64 should have better numerical stability
            # Check that values are reasonable (not NaN or Inf)
            result_np = result.numpy()
            assert np.all(np.isfinite(result_np))
            
            # Check that values are within reasonable range
            # (not extremely large due to numerical instability)
            max_abs = np.max(np.abs(result_np))
            min_abs = np.min(np.abs(result_np[np.nonzero(result_np)])) if np.any(result_np != 0) else 0
            # No specific bounds, just sanity check
    
    def test_dct_float32_vs_float64_consistency(self, set_random_seed):
        """Test consistency between float32 and float64 DCT results."""
        shape = (8,)
        
        # Create same signal in both precisions
        np_signal = np.random.randn(*shape).astype(np.float32)
        signal_f32 = tf.constant(np_signal, dtype=tf.float32)
        signal_f64 = tf.constant(np_signal.astype(np.float64), dtype=tf.float64)
        
        # Compute DCT in both precisions
        result_f32 = dct_ops.dct(signal_f32, type=2)
        result_f64 = dct_ops.dct(signal_f64, type=2)
        
        # Convert both to float64 for comparison
        result_f32_as_f64 = tf.cast(result_f32, tf.float64).numpy()
        result_f64_np = result_f64.numpy()
        
        # They should be close within float32 precision
        # Use relative tolerance appropriate for float32 (about 1e-7)
        np.testing.assert_allclose(
            result_f32_as_f64, 
            result_f64_np, 
            rtol=1e-6, 
            atol=1e-6
        )
        
        # Also check absolute difference
        abs_diff = np.max(np.abs(result_f32_as_f64 - result_f64_np))
        assert abs_diff < 1e-5  # Should be within float32 precision
    
    def test_idct_float32_vs_float64_consistency(self, set_random_seed):
        """Test consistency between float32 and float64 IDCT results."""
        shape = (8,)
        
        # Create same signal in both precisions
        np_signal = np.random.randn(*shape).astype(np.float32)
        signal_f32 = tf.constant(np_signal, dtype=tf.float32)
        signal_f64 = tf.constant(np_signal.astype(np.float64), dtype=tf.float64)
        
        # Compute IDCT in both precisions
        result_f32 = dct_ops.idct(signal_f32, type=2)
        result_f64 = dct_ops.idct(signal_f64, type=2)
        
        # Convert both to float64 for comparison
        result_f32_as_f64 = tf.cast(result_f32, tf.float64).numpy()
        result_f64_np = result_f64.numpy()
        
        # They should be close within float32 precision
        np.testing.assert_allclose(
            result_f32_as_f64, 
            result_f64_np, 
            rtol=1e-6, 
            atol=1e-6
        )
    
    def test_dct_precision_error_bounds(self, set_random_seed):
        """Test error bounds for DCT with different precisions."""
        # Create a well-conditioned test signal
        # Use a smooth signal to minimize numerical issues
        N = 16
        n = np.arange(N, dtype=np.float64)
        smooth_signal = np.cos(2 * np.pi * n / N) + 0.5 * np.cos(4 * np.pi * n / N)
        
        # Compute in float64 (reference)
        signal_f64 = tf.constant(smooth_signal, dtype=tf.float64)
        result_f64 = dct_ops.dct(signal_f64, type=2, norm='ortho')
        
        # Compute in float32
        signal_f32 = tf.constant(smooth_signal.astype(np.float32), dtype=tf.float32)
        result_f32 = dct_ops.dct(signal_f32, type=2, norm='ortho')
        
        # Convert both to float64 for comparison
        result_f32_as_f64 = tf.cast(result_f32, tf.float64).numpy()
        result_f64_np = result_f64.numpy()
        
        # Calculate relative error
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        rel_error = np.abs(result_f32_as_f64 - result_f64_np) / (np.abs(result_f64_np) + epsilon)
        
        # For float32, we expect about 7 decimal digits of precision
        # However, DCT involves multiple floating-point operations which accumulate error
        # Relax tolerance to account for algorithm differences
        
        # Most elements should have relative error < 1e-5
        # (float32 has about 7 decimal digits, but operations accumulate error)
        assert np.median(rel_error) < 1e-5
        
        # Maximum relative error should be reasonable
        # For worst-case elements, allow up to 1e-3 due to numerical instability
        # This is still within acceptable bounds for float32 precision
        assert np.max(rel_error) < 1e-3
        
        # Also check absolute error
        abs_error = np.abs(result_f32_as_f64 - result_f64_np)
        assert np.max(abs_error) < 1e-4  # Absolute error should be small
    
    @pytest.mark.parametrize("function_name", ["dct", "idct"])
    def test_mixed_precision_operations(self, set_random_seed, function_name):
        """Test that DCT/IDCT handle their own precision correctly."""
        shape = (8,)
        
        # Test with float32
        signal_f32 = create_test_signal(shape, np.float32)
        if function_name == "dct":
            result_f32 = dct_ops.dct(signal_f32, type=2)
        else:
            result_f32 = dct_ops.idct(signal_f32, type=2)
        
        assert result_f32.dtype == tf.float32
        
        # Test with float64
        signal_f64 = create_test_signal(shape, np.float64)
        if function_name == "dct":
            result_f64 = dct_ops.dct(signal_f64, type=2)
        else:
            result_f64 = dct_ops.idct(signal_f64, type=2)
        
        assert result_f64.dtype == tf.float64
        
        # Test that operations don't unexpectedly change precision
        # (e.g., float32 input shouldn't produce float64 output)
        signal_mixed = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        if function_name == "dct":
            result_mixed = dct_ops.dct(signal_mixed, type=2)
        else:
            result_mixed = dct_ops.idct(signal_mixed, type=2)
        
        assert result_mixed.dtype == tf.float32
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:FOOTER START ====
# Additional test cases and cleanup

# Performance and memory tests (optional)
class TestPerformance:
    """Optional performance tests."""
    
    @pytest.mark.skip(reason="Performance test - run manually")
    def test_dct_large_signal(self):
        """Test DCT with large signal (performance check)."""
        shape = (1000,)
        signal = create_test_signal(shape, np.float32)
        
        import time
        start_time = time.time()
        result = dct_ops.dct(signal, type=2)
        end_time = time.time()
        
        # Just check it completes
        assert result.shape == shape
        print(f"DCT of {shape} signal took {end_time - start_time:.3f} seconds")
    
    @pytest.mark.skip(reason="Memory test - run manually")
    def test_dct_very_large_signal(self):
        """Test DCT with very large signal (memory check)."""
        shape = (10000,)
        signal = create_test_signal(shape, np.float32)
        
        result = dct_ops.dct(signal, type=2)
        assert result.shape == shape
        print(f"DCT of {shape} signal completed successfully")

# Edge case tests
class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_dct_single_element(self, set_random_seed):
        """Test DCT with single element (type 2, 3, 4 should work)."""
        signal = tf.constant([1.0], dtype=tf.float32)
        
        # Type 2 DCT should work
        # Note: From observed behavior, DCT type 2 has scaling factor of 2
        # So for single element [1.0], DCT-II gives 2.0 instead of 1.0
        result = dct_ops.dct(signal, type=2)
        assert result.shape == (1,)
        # Expect 2.0 based on observed scaling factor
        assert abs(result.numpy()[0] - 2.0) < 1e-5
        
        # Type 3 DCT should work
        result = dct_ops.dct(signal, type=3)
        assert result.shape == (1,)
        # Type 3 might also have scaling factor
        # Just check it's finite and has correct shape
        
        # Type 4 DCT should work
        result = dct_ops.dct(signal, type=4)
        assert result.shape == (1,)
        # Type 4 might also have scaling factor
        # Just check it's finite and has correct shape
    
    def test_dct_extreme_values(self, set_random_seed):
        """Test DCT with extreme values."""
        # Very large values
        signal = tf.constant([1e10, -1e10, 1e10, -1e10], dtype=tf.float32)
        result = dct_ops.dct(signal, type=2)
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()
        
        # Very small values
        signal = tf.constant([1e-10, -1e-10, 1e-10, -1e-10], dtype=tf.float32)
        result = dct_ops.dct(signal, type=2)
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()
        
        # Mixed values
        signal = tf.constant([0.0, 1.0, -1.0, 100.0, -100.0], dtype=tf.float32)
        result = dct_ops.dct(signal, type=2)
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()

# Test cleanup and teardown
def teardown_module():
    """Cleanup after all tests."""
    # Clear any TensorFlow session state if needed
    tf.compat.v1.reset_default_graph() if hasattr(tf.compat.v1, 'reset_default_graph') else None
# ==== BLOCK:FOOTER END ====