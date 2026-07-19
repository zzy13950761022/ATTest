import torch
import pytest
import numpy as np
from torch.utils.checkpoint import checkpoint

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def simple_linear(x, weight=None, bias=None):
    """Simple linear function for testing."""
    if weight is None:
        weight = torch.ones_like(x)
    if bias is None:
        bias = torch.zeros_like(x)
    return x * weight + bias

def random_operation(x, seed=42):
    """Function with random operations for RNG testing."""
    torch.manual_seed(seed)
    return x + torch.randn_like(x) * 0.1

def nested_output(x):
    """Function returning nested structure."""
    return {
        'tensor': x,
        'list': [x * 2, x * 3],
        'tuple': (x + 1, x - 1),
        'scalar': 42,
        'nested': {'inner': x * 0.5}
    }

def invalid_callable():
    """Invalid callable that raises RuntimeError."""
    raise RuntimeError("Invalid function call")

def approx_equal(a, b, rtol=1e-5, atol=1e-8):
    """Check if two tensors are approximately equal."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.allclose(a, b, rtol=rtol, atol=atol)
    return False

def check_output_structure(output1, output2):
    """Check if two nested structures have same structure and tensor values."""
    if isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
        return approx_equal(output1, output2)
    elif isinstance(output1, (list, tuple)) and isinstance(output2, (list, tuple)):
        if len(output1) != len(output2):
            return False
        return all(check_output_structure(o1, o2) for o1, o2 in zip(output1, output2))
    elif isinstance(output1, dict) and isinstance(output2, dict):
        if set(output1.keys()) != set(output2.keys()):
            return False
        return all(check_output_structure(output1[k], output2[k]) for k in output1.keys())
    else:
        # For non-tensor values, check equality
        return output1 == output2

@pytest.fixture
def random_tensor():
    """Fixture providing random tensor for tests."""
    set_random_seed(42)
    return torch.randn(2, 3, dtype=torch.float32)

@pytest.fixture
def requires_grad_tensor():
    """Fixture providing tensor with requires_grad=True."""
    set_random_seed(42)
    x = torch.randn(2, 3, dtype=torch.float32)
    x.requires_grad_(True)
    return x
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# TC-01: 基础检查点功能验证
@pytest.mark.parametrize("function_type,input_shape,dtype,device,use_reentrant,preserve_rng_state", [
    # Base case from test plan
    ("simple_linear", [2, 3], "float32", "cpu", True, True),
    # Parameter extensions
    ("simple_linear", [4, 4], "float64", "cpu", True, True),
    ("simple_linear", [2, 3], "float32", "cuda", True, True),
])
def test_checkpoint_basic_functionality(
    function_type, input_shape, dtype, device, use_reentrant, preserve_rng_state
):
    """
    Test basic checkpoint functionality with various configurations.
    """
    # Skip CUDA tests if CUDA not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create input tensor
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float64":
        torch_dtype = torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    x = torch.randn(*input_shape, dtype=torch_dtype)
    if device == "cuda":
        x = x.cuda()
    
    # Define function based on type
    if function_type == "simple_linear":
        def func(tensor):
            return simple_linear(tensor)
    else:
        raise ValueError(f"Unsupported function_type: {function_type}")
    
    # Compute direct result
    direct_result = func(x)
    
    # Compute checkpoint result
    if preserve_rng_state:
        checkpoint_result = checkpoint(
            func, x, 
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve_rng_state
        )
    else:
        checkpoint_result = checkpoint(
            func, x,
            use_reentrant=use_reentrant
        )
    
    # WEAK ASSERTIONS (epoch 1)
    # 1. Output shape matches
    assert checkpoint_result.shape == direct_result.shape, \
        f"Checkpoint output shape {checkpoint_result.shape} != direct shape {direct_result.shape}"
    
    # 2. Output dtype matches
    assert checkpoint_result.dtype == direct_result.dtype, \
        f"Checkpoint output dtype {checkpoint_result.dtype} != direct dtype {direct_result.dtype}"
    
    # 3. Output values are finite
    assert torch.isfinite(checkpoint_result).all(), \
        "Checkpoint output contains non-finite values"
    
    # 4. No exception raised (implicitly passed if we get here)
    
    # Note: Strong assertions (approx_equal_direct, gradient_correctness, memory_usage_improved)
    # will be added in final round when assertion_level is "strong"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# TC-02: use_reentrant模式参数验证
@pytest.mark.parametrize("function_type,input_shape,dtype,device,use_reentrant,preserve_rng_state,has_kwargs", [
    # Base case from test plan
    ("simple_linear", [2, 3], "float32", "cpu", False, True, True),
    # Parameter extension
    ("simple_linear", [2, 3], "float32", "cpu", False, False, True),
])
def test_checkpoint_use_reentrant_modes(
    function_type, input_shape, dtype, device, use_reentrant, preserve_rng_state, has_kwargs
):
    """
    Test checkpoint with use_reentrant=False mode and keyword argument support.
    """
    # Skip CUDA tests if CUDA not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create input tensor
    if dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    x = torch.randn(*input_shape, dtype=torch_dtype)
    if device == "cuda":
        x = x.cuda()
    
    # Define function with optional kwargs
    if function_type == "simple_linear":
        if has_kwargs:
            def func(tensor, weight=None, bias=None):
                if weight is None:
                    weight = torch.ones_like(tensor)
                if bias is None:
                    bias = torch.zeros_like(tensor)
                return tensor * weight + bias
        else:
            def func(tensor):
                return simple_linear(tensor)
    else:
        raise ValueError(f"Unsupported function_type: {function_type}")
    
    # Compute direct result
    if has_kwargs:
        weight = torch.ones_like(x) * 2.0
        bias = torch.ones_like(x) * 0.5
        direct_result = func(x, weight=weight, bias=bias)
    else:
        direct_result = func(x)
    
    # Compute checkpoint result
    if use_reentrant:
        # use_reentrant=True doesn't support kwargs
        if has_kwargs:
            # This should raise RuntimeError, but we're testing valid cases
            # For this test, we'll skip kwargs when use_reentrant=True
            checkpoint_result = checkpoint(func, x, use_reentrant=use_reentrant)
        else:
            checkpoint_result = checkpoint(func, x, use_reentrant=use_reentrant)
    else:
        # use_reentrant=False supports kwargs
        if has_kwargs:
            checkpoint_result = checkpoint(
                func, x,
                weight=weight, bias=bias,
                use_reentrant=use_reentrant,
                preserve_rng_state=preserve_rng_state
            )
        else:
            checkpoint_result = checkpoint(
                func, x,
                use_reentrant=use_reentrant,
                preserve_rng_state=preserve_rng_state
            )
    
    # WEAK ASSERTIONS (epoch 1)
    # 1. Output shape matches
    assert checkpoint_result.shape == direct_result.shape, \
        f"Checkpoint output shape {checkpoint_result.shape} != direct shape {direct_result.shape}"
    
    # 2. Output dtype matches
    assert checkpoint_result.dtype == direct_result.dtype, \
        f"Checkpoint output dtype {checkpoint_result.dtype} != direct dtype {direct_result.dtype}"
    
    # 3. kwargs supported when use_reentrant=False
    if not use_reentrant and has_kwargs:
        # Verify that kwargs were actually used by checking the result
        # Since weight=2.0 and bias=0.5, result should be x*2 + 0.5
        expected = x * 2.0 + 0.5
        assert torch.allclose(checkpoint_result, expected, rtol=1e-5, atol=1e-8), \
            "kwargs not properly passed to checkpointed function"
    
    # 4. No exception raised (implicitly passed if we get here)
    
    # Note: Strong assertions (approx_equal_direct, gradient_correctness, kwargs_preserved)
    # will be added in final round when assertion_level is "strong"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: 梯度正确性验证
@pytest.mark.parametrize("function_type,input_shape,dtype,device,use_reentrant,preserve_rng_state,requires_grad", [
    # Base case from test plan
    ("simple_linear", [2, 3], "float32", "cpu", True, True, True),
    # Parameter extension
    ("simple_linear", [2, 3], "float32", "cpu", False, True, True),
])
def test_checkpoint_gradient_correctness(
    function_type, input_shape, dtype, device, use_reentrant, preserve_rng_state, requires_grad
):
    """
    Test gradient correctness of checkpointed functions.
    """
    # Skip CUDA tests if CUDA not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create input tensor with requires_grad
    if dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    x = torch.randn(*input_shape, dtype=torch_dtype)
    if device == "cuda":
        x = x.cuda()
    
    if requires_grad:
        x.requires_grad_(True)
    
    # Define function
    if function_type == "simple_linear":
        def func(tensor):
            return simple_linear(tensor).sum()  # Sum to get scalar for gradient
    else:
        raise ValueError(f"Unsupported function_type: {function_type}")
    
    # Compute direct gradient
    direct_output = func(x)
    direct_output.backward()
    direct_grad = x.grad.clone() if x.grad is not None else None
    x.grad = None  # Reset gradient
    
    # Compute checkpoint gradient
    if preserve_rng_state:
        checkpoint_output = checkpoint(
            func, x,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve_rng_state
        )
    else:
        checkpoint_output = checkpoint(
            func, x,
            use_reentrant=use_reentrant
        )
    
    checkpoint_output.backward()
    checkpoint_grad = x.grad.clone() if x.grad is not None else None
    
    # WEAK ASSERTIONS (epoch 1)
    # 1. Gradient exists
    assert checkpoint_grad is not None, "Checkpoint gradient is None"
    assert direct_grad is not None, "Direct gradient is None"
    
    # 2. Gradient shape matches input shape
    assert checkpoint_grad.shape == x.shape, \
        f"Checkpoint gradient shape {checkpoint_grad.shape} != input shape {x.shape}"
    assert direct_grad.shape == x.shape, \
        f"Direct gradient shape {direct_grad.shape} != input shape {x.shape}"
    
    # 3. Gradient values are finite
    assert torch.isfinite(checkpoint_grad).all(), \
        "Checkpoint gradient contains non-finite values"
    assert torch.isfinite(direct_grad).all(), \
        "Direct gradient contains non-finite values"
    
    # 4. No exception raised (implicitly passed if we get here)
    
    # Note: Strong assertions (gradient_approx_equal, gradient_precision, backward_compatibility)
    # will be added in final round when assertion_level is "strong"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: 异常场景处理
@pytest.mark.parametrize("function_type,use_reentrant,has_kwargs,expected_exception", [
    # Base case from test plan
    ("invalid_callable", True, True, "RuntimeError"),
])
def test_checkpoint_exception_handling(
    function_type, use_reentrant, has_kwargs, expected_exception
):
    """
    Test exception handling in checkpointed functions.
    """
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create a simple input tensor
    x = torch.randn(2, 3, dtype=torch.float32)
    
    # Define function based on type
    if function_type == "invalid_callable":
        # Create a callable that raises RuntimeError
        def invalid_func(tensor, **kwargs):
            raise RuntimeError("Invalid function call - test exception")
    else:
        raise ValueError(f"Unsupported function_type: {function_type}")
    
    # Mock targets as specified in test plan
    from unittest.mock import patch, MagicMock
    
    # Test different exception scenarios based on use_reentrant
    if use_reentrant:
        # Test with use_reentrant=True
        if has_kwargs:
            # use_reentrant=True doesn't support kwargs, should raise ValueError
            # before the function is even called
            with pytest.raises(ValueError) as exc_info:
                checkpoint(invalid_func, x, use_reentrant=use_reentrant, extra_kwarg=1)
            
            # WEAK ASSERTIONS (epoch 4)
            # 1. Exception raised
            assert exc_info.type == ValueError, f"Expected ValueError for kwargs with use_reentrant=True, got {exc_info.type}"
            
            # 2. Exception type matches
            # Already checked above
            
            # 3. Exception message contains expected text
            error_msg = str(exc_info.value).lower()
            # Check for keyword arguments error message
            assert "unexpected keyword" in error_msg or "keyword" in error_msg, \
                f"Error message should mention keyword arguments. Got: {error_msg}"
            
            # Also test that the function is not called when kwargs are provided with use_reentrant=True
            # We can verify this by checking that our RuntimeError is not raised
        else:
            # Without kwargs, invalid function should raise RuntimeError
            # Mock CheckpointFunction.apply to test exception handling
            with patch('torch.utils.checkpoint.CheckpointFunction.apply') as mock_apply:
                mock_apply.side_effect = RuntimeError("Mocked CheckpointFunction error")
                
                with pytest.raises(RuntimeError) as exc_info:
                    checkpoint(invalid_func, x, use_reentrant=use_reentrant)
                
                # WEAK ASSERTIONS
                # 1. Exception raised
                assert exc_info.type == RuntimeError, f"Expected RuntimeError, got {exc_info.type}"
                
                # 2. Exception type matches
                # Already checked above
                
                # 3. Exception message contains expected text
                error_msg = str(exc_info.value)
                assert "Mocked CheckpointFunction error" in error_msg, \
                    f"Error message should contain mocked error. Got: {error_msg}"
                
                # Verify the mock was called
                assert mock_apply.called, "CheckpointFunction.apply should be called"
    else:
        # Test with use_reentrant=False
        if has_kwargs:
            # use_reentrant=False supports kwargs, so the function should be called
            # and raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                checkpoint(invalid_func, x, use_reentrant=use_reentrant, extra_kwarg=1)
            
            # WEAK ASSERTIONS
            # 1. Exception raised
            assert exc_info.type == RuntimeError, f"Expected RuntimeError, got {exc_info.type}"
            
            # 2. Exception type matches
            # Already checked above
            
            # 3. Exception message contains expected text
            error_msg = str(exc_info.value)
            # Should contain our test exception message
            assert "Invalid function call" in error_msg or "test exception" in error_msg, \
                f"Error message should contain test exception. Got: {error_msg}"
        else:
            # Without kwargs, invalid function should raise RuntimeError
            # Mock _checkpoint_without_reentrant to test exception handling
            with patch('torch.utils.checkpoint._checkpoint_without_reentrant') as mock_checkpoint:
                mock_checkpoint.side_effect = RuntimeError("Mocked checkpoint error")
                
                with pytest.raises(RuntimeError) as exc_info:
                    checkpoint(invalid_func, x, use_reentrant=use_reentrant)
                
                # WEAK ASSERTIONS
                # 1. Exception raised
                assert exc_info.type == RuntimeError, f"Expected RuntimeError, got {exc_info.type}"
                
                # 2. Exception type matches
                # Already checked above
                
                # 3. Exception message contains expected text
                error_msg = str(exc_info.value)
                assert "Mocked checkpoint error" in error_msg, \
                    f"Error message should contain mocked error. Got: {error_msg}"
                
                # Verify the mock was called
                assert mock_checkpoint.called, "_checkpoint_without_reentrant should be called"
    
    # Additional test: Test that a non-callable function raises TypeError
    with pytest.raises(TypeError) as exc_info:
        checkpoint("not a function", x, use_reentrant=use_reentrant)
    
    # Verify TypeError is raised for non-callable
    assert exc_info.type == TypeError, f"Expected TypeError for non-callable, got {exc_info.type}"
    
    # Note: Strong assertions (exception_context, error_recovery)
    # will be added in final round when assertion_level is "strong"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: RNG状态管理验证 (DEFERRED - placeholder)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# TC-06: 嵌套输出结构处理 (DEFERRED - placeholder)
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:FOOTER START ====
# Additional test utilities and cleanup

if __name__ == "__main__":
    # Simple test runner for debugging
    import sys
    pytest.main([__file__] + sys.argv[1:])
# ==== BLOCK:FOOTER END ====