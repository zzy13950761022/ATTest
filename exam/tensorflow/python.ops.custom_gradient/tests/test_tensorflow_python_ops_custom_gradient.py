"""
Test cases for tensorflow.python.ops.custom_gradient
"""
import math
import numpy as np
import pytest
import tensorflow as tf
from unittest import mock

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ==== BLOCK:HEADER START ====
# Helper functions
def create_test_tensor(shape, dtype=tf.float32):
    """Create test tensor with random values."""
    if dtype == tf.float32:
        return tf.random.uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
    elif dtype == tf.float64:
        return tf.random.uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def assert_tensors_close(a, b, rtol=1e-6, atol=1e-6, msg=None):
    """Assert two tensors are close within tolerance."""
    if isinstance(a, tf.Tensor) and isinstance(b, tf.Tensor):
        a_np = a.numpy() if hasattr(a, 'numpy') else a
        b_np = b.numpy() if hasattr(b, 'numpy') else b
        np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol, err_msg=msg)
    elif isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        # For scalar numeric values
        if msg:
            assert abs(a - b) <= atol + rtol * abs(b), msg
        else:
            assert abs(a - b) <= atol + rtol * abs(b)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        # For numpy arrays
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)
    else:
        # For other types, use equality
        if msg:
            assert a == b, msg
        else:
            assert a == b

def simple_operation(x):
    """Simple operation for testing: f(x) = x^2, grad = 2x."""
    def grad_fn(upstream):
        return upstream * 2 * x
    return x * x, grad_fn

def square_operation(x):
    """Square operation: f(x) = x^2, grad = 2x."""
    def grad_fn(upstream):
        return upstream * 2 * x
    return x * x, grad_fn

def linear_with_variable(x, variable):
    """Linear operation with variable: f(x) = x * variable."""
    def grad_fn(upstream):
        # For single variable case, return gradient for x only
        return upstream * variable
    return x * variable, grad_fn

def nested_operation(x):
    """Nested operation for testing composition."""
    def grad_fn(upstream):
        return upstream * 3 * x * x
    return x * x * x, grad_fn

def log1pexp(x):
    """Numerically stable log1pexp implementation."""
    e = tf.exp(x)
    def grad_fn(upstream):
        return upstream * (1 - 1 / (1 + e))
    return tf.math.log(1 + e), grad_fn

# Fixtures
@pytest.fixture
def test_tensor_2x2_float32():
    """Fixture for 2x2 float32 tensor."""
    return create_test_tensor([2, 2], tf.float32)

@pytest.fixture
def test_tensor_3x3_float32():
    """Fixture for 3x3 float32 tensor."""
    return create_test_tensor([3, 3], tf.float32)

@pytest.fixture
def test_variable_2x2_float32():
    """Fixture for 2x2 float32 variable."""
    return tf.Variable(create_test_tensor([2, 2], tf.float32))

@pytest.fixture
def test_variable_3x3_float32():
    """Fixture for 3x3 float32 variable."""
    return tf.Variable(create_test_tensor([3, 3], tf.float32))
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize("mode,dtype,shape,function_type", [
    ("eager", tf.float32, [2, 2], "simple_operation"),
    ("graph", tf.float64, [4, 4], "simple_operation"),
    ("eager", tf.float32, [1, 10], "simple_operation"),
])
def test_basic_decorator_functionality(mode, dtype, shape, function_type):
    """TC-01: 基本装饰器功能验证"""
    # Skip if mode not supported
    if mode == "graph" and tf.executing_eagerly():
        pytest.skip("Graph mode test requires non-eager execution")
    
    # Create test tensor
    x = create_test_tensor(shape, dtype)
    
    # Define the operation based on function_type
    if function_type == "simple_operation":
        @tf.custom_gradient
        def decorated_func(x):
            return simple_operation(x)
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    
    # Test in appropriate mode
    if mode == "graph":
        # Graph mode execution
        @tf.function
        def compute_gradient(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = decorated_func(x)
            return y, tape.gradient(y, x)
        
        y, grad = compute_gradient(x)
    else:
        # Eager mode execution
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = decorated_func(x)
        grad = tape.gradient(y, x)
    
    # Weak assertions (epoch 1)
    # 1. output_shape
    assert y.shape == shape, f"Output shape mismatch: expected {shape}, got {y.shape}"
    
    # 2. output_dtype
    assert y.dtype == dtype, f"Output dtype mismatch: expected {dtype}, got {y.dtype}"
    
    # 3. gradient_exists
    assert grad is not None, "Gradient should exist"
    
    # 4. basic_gradient_correctness
    # For simple_operation: f(x) = x^2, grad = 2x
    expected_grad = 2 * x
    assert_tensors_close(grad, expected_grad, rtol=1e-6, atol=1e-6)
    
    # Verify output value matches expected
    expected_y = x * x
    assert_tensors_close(y, expected_y, rtol=1e-6, atol=1e-6)
    
    # Additional verification: gradient shape matches input shape
    assert grad.shape == x.shape, f"Gradient shape mismatch: expected {x.shape}, got {grad.shape}"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("mode,dtype,shape,function_type", [
    ("both", tf.float32, [3, 3], "square_operation"),
    ("both", tf.float64, [5, 5], "square_operation"),
])
def test_graph_eager_mode_consistency(mode, dtype, shape, function_type):
    """TC-02: 图模式与急切模式一致性"""
    # Create test tensor
    x = create_test_tensor(shape, dtype)
    
    # Define the operation
    if function_type == "square_operation":
        @tf.custom_gradient
        def decorated_func(x):
            return square_operation(x)
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    
    # Mock targets as specified in test_plan
    # Updated paths for TensorFlow 2.x compatibility
    # Note: In TF 2.x, these internal APIs may not be directly accessible
    # We'll mock them but also handle the case where they might not exist
    try:
        # Try to import the modules to check if they exist
        from tensorflow.python.framework import ops as tf_ops
        from tensorflow.python.eager import tape as tf_tape
        
        # Mock the specific functions
        with mock.patch.object(tf_ops, 'RegisterGradient') as mock_register_gradient, \
             mock.patch.object(tf_tape, 'record_operation') as mock_record_operation:
            
            # Configure mocks to do nothing (just track calls)
            mock_register_gradient.return_value = lambda f: f
            mock_record_operation.return_value = None
            
            # Run the actual test logic
            _run_mode_consistency_test(x, decorated_func, mode)
            
    except ImportError:
        # If internal modules are not accessible, run test without mocks
        # This is common in TF 2.x where internal APIs are not exposed
        _run_mode_consistency_test(x, decorated_func, mode)

def _run_mode_consistency_test(x, decorated_func, mode):
    """Helper function to run mode consistency test without mocks."""
    # Eager mode computation
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_eager = decorated_func(x)
    grad_eager = tape.gradient(y_eager, x)
    
    # Graph mode computation
    @tf.function
    def compute_in_graph(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = decorated_func(x)
        return y, tape.gradient(y, x)
    
    y_graph, grad_graph = compute_in_graph(x)
    
    # Weak assertions (epoch 1)
    # 1. mode_consistency_output
    assert_tensors_close(y_eager, y_graph, rtol=1e-6, atol=1e-6,
                        msg="Output mismatch between eager and graph modes")
    
    # 2. mode_consistency_gradient
    assert_tensors_close(grad_eager, grad_graph, rtol=1e-6, atol=1e-6,
                        msg="Gradient mismatch between eager and graph modes")
    
    # 3. graph_mode_support
    assert y_graph is not None, "Graph mode should produce output"
    assert grad_graph is not None, "Graph mode should produce gradient"
    
    # 4. eager_mode_support
    assert y_eager is not None, "Eager mode should produce output"
    assert grad_eager is not None, "Eager mode should produce gradient"
    
    # Additional verification: correctness of computation
    expected_y = x * x
    expected_grad = 2 * x
    
    assert_tensors_close(y_eager, expected_y, rtol=1e-6, atol=1e-6)
    assert_tensors_close(grad_eager, expected_grad, rtol=1e-6, atol=1e-6)
    assert_tensors_close(y_graph, expected_y, rtol=1e-6, atol=1e-6)
    assert_tensors_close(grad_graph, expected_grad, rtol=1e-6, atol=1e-6)
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize("mode,dtype,shape,variable_count,function_type", [
    ("eager", tf.float32, [2, 2], 1, "linear_with_variable"),
    ("eager", tf.float32, [3, 3], 2, "linear_with_variable"),
])
def test_resource_variable_gradient_propagation(mode, dtype, shape, variable_count, function_type):
    """TC-03: ResourceVariable梯度传播"""
    # Skip if mode not supported
    if mode == "graph" and tf.executing_eagerly():
        pytest.skip("Graph mode test requires non-eager execution")
    
    # Create test tensor and variables
    x = create_test_tensor(shape, dtype)
    
    if variable_count == 1:
        variables = [tf.Variable(create_test_tensor(shape, dtype))]
    elif variable_count == 2:
        variables = [
            tf.Variable(create_test_tensor(shape, dtype)),
            tf.Variable(create_test_tensor(shape, dtype))
        ]
    else:
        raise ValueError(f"Unsupported variable_count: {variable_count}")
    
    # Define the operation
    if function_type == "linear_with_variable":
        if variable_count == 1:
            @tf.custom_gradient
            def decorated_func(x):
                # Use linear_with_variable function which already has correct grad_fn
                return linear_with_variable(x, variables[0])
        else:
            # For multiple variables, we need a custom implementation
            @tf.custom_gradient
            def decorated_func(x):
                # Use first variable for computation
                result = x * variables[0]
                
                def grad_fn(upstream):
                    # Return gradients for x and all variables
                    grad_x = upstream * variables[0]
                    # Gradient for first variable
                    grad_v0 = upstream * x
                    # Gradients for other variables (zero since not used)
                    grad_vars = [grad_v0] + [tf.zeros_like(v) for v in variables[1:]]
                    return (grad_x, grad_vars)
                
                return result, grad_fn
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    
    # Mock targets as specified in test_plan
    # Note: In TF 2.x, gradients_util module may not be directly accessible
    # We'll create simple mock functions that simulate the behavior
    try:
        # Try to import the custom_gradient module to check what's available
        from tensorflow.python.ops import custom_gradient as tf_custom_gradient
        
        # Check if the functions exist in the module
        has_get_dependent = hasattr(tf_custom_gradient, '_get_dependent_variables')
        has_get_by_name = hasattr(tf_custom_gradient, 'get_variable_by_name')
        
        if has_get_dependent and has_get_by_name:
            # Mock the actual functions if they exist
            with mock.patch.object(tf_custom_gradient, '_get_dependent_variables') as mock_get_dependent, \
                 mock.patch.object(tf_custom_gradient, 'get_variable_by_name') as mock_get_by_name:
                
                # Configure mocks
                mock_get_dependent.return_value = variables
                mock_get_by_name.return_value = variables[0] if variable_count == 1 else variables
                
                # Run test with mocks
                _run_variable_gradient_test(x, variables, decorated_func, variable_count)
        else:
            # Functions don't exist, run test without mocks
            _run_variable_gradient_test(x, variables, decorated_func, variable_count)
            
    except ImportError:
        # If module is not accessible, run test without mocks
        _run_variable_gradient_test(x, variables, decorated_func, variable_count)

def _run_variable_gradient_test(x, variables, decorated_func, variable_count):
    """Helper function to run variable gradient test without mocks."""
    # Compute gradients
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        for var in variables:
            tape.watch(var)
        y = decorated_func(x)
    
    # Compute gradients
    grad_x = tape.gradient(y, x)
    grad_vars = [tape.gradient(y, var) for var in variables]
    
    # Weak assertions (epoch 1)
    # 1. variable_gradient_exists
    assert grad_x is not None, "Input gradient should exist"
    for i, grad_var in enumerate(grad_vars):
        assert grad_var is not None, f"Variable {i} gradient should exist"
    
    # 2. variable_gradient_shape
    assert grad_x.shape == x.shape, f"Input gradient shape mismatch: expected {x.shape}, got {grad_x.shape}"
    for i, (var, grad_var) in enumerate(zip(variables, grad_vars)):
        assert grad_var.shape == var.shape, f"Variable {i} gradient shape mismatch: expected {var.shape}, got {grad_var.shape}"
    
    # 3. resource_variable_type
    for i, var in enumerate(variables):
        assert isinstance(var, tf.Variable), f"Variable {i} should be a tf.Variable"
        # Check if it's a ResourceVariable (default in TF2)
        assert hasattr(var, 'handle'), f"Variable {i} should be a ResourceVariable"
    
    # 4. gradient_propagation
    # For linear_with_variable: f(x) = x * v, grad_x = v, grad_v = x
    expected_grad_x = variables[0]
    assert_tensors_close(grad_x, expected_grad_x, rtol=1e-6, atol=1e-6)
    
    if variable_count >= 1:
        expected_grad_v0 = x
        assert_tensors_close(grad_vars[0], expected_grad_v0, rtol=1e-6, atol=1e-6)
    
    # Verify output value
    expected_y = x * variables[0]
    assert_tensors_close(y, expected_y, rtol=1e-6, atol=1e-6)
    
    # For multiple variables, additional variables should have zero gradient
    # (since they're not used in the computation)
    for i in range(1, variable_count):
        expected_zero = tf.zeros_like(variables[i])
        assert_tensors_close(grad_vars[i], expected_zero, rtol=1e-6, atol=1e-6)
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
@pytest.mark.parametrize("mode,dtype,shape,nesting_level,function_type", [
    ("eager", tf.float32, [2, 2], 2, "nested_operation"),
    ("eager", tf.float64, [3, 3], 3, "nested_operation"),
])
def test_nested_custom_gradient_scenarios(mode, dtype, shape, nesting_level, function_type):
    """TC-04: 嵌套自定义梯度场景"""
    # Skip if mode not supported
    if mode == "graph" and tf.executing_eagerly():
        pytest.skip("Graph mode test requires non-eager execution")
    
    # Create test tensor
    x = create_test_tensor(shape, dtype)
    
    # Define nested operations
    if function_type == "nested_operation":
        # Create nested custom gradient functions
        def create_nested_function(level):
            if level == 1:
                @tf.custom_gradient
                def inner_func(x):
                    # f(x) = x^3, grad = 3x^2
                    result = x * x * x
                    def grad_fn(upstream):
                        return upstream * 3 * x * x
                    return result, grad_fn
                return inner_func
            else:
                @tf.custom_gradient
                def outer_func(x):
                    # Compose functions: f_n(f_{n-1}(...f_1(x)...))
                    inner = create_nested_function(level - 1)
                    result = inner(x)
                    
                    def grad_fn(upstream):
                        # Chain rule: gradient through composition
                        # For f(g(x)), grad = f'(g(x)) * g'(x)
                        # We'll compute this numerically using GradientTape
                        with tf.GradientTape() as tape:
                            tape.watch(x)
                            y = inner(x)
                        grad_inner = tape.gradient(y, x)
                        return upstream * grad_inner
                    
                    return result, grad_fn
                return outer_func
        
        # Create the nested function
        nested_func = create_nested_function(nesting_level)
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    
    # Weak assertions (epoch 1-4)
    # 1. nested_output_correct
    y = nested_func(x)
    expected_y = x ** (3 ** nesting_level)  # x^(3^n)
    assert_tensors_close(y, expected_y, rtol=1e-6, atol=1e-6)
    
    # 2. nested_gradient_exists
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = nested_func(x)
    grad_x = tape.gradient(y, x)
    assert grad_x is not None, "Gradient should exist for nested function"
    
    # 3. composition_works
    # Verify that composition doesn't break gradient computation
    assert grad_x.shape == x.shape, f"Gradient shape mismatch: expected {x.shape}, got {grad_x.shape}"
    
    # 4. no_side_effects
    # Run function multiple times to ensure no side effects
    y1 = nested_func(x)
    y2 = nested_func(x)
    assert_tensors_close(y1, y2, rtol=1e-6, atol=1e-6)
    
    # Strong assertions (epoch 5)
    # 1. nested_gradient_correctness
    # Compute gradient analytically: d/dx [x^(3^n)] = (3^n) * x^(3^n - 1)
    power = 3 ** nesting_level
    expected_grad = power * (x ** (power - 1))
    assert_tensors_close(grad_x, expected_grad, rtol=1e-6, atol=1e-6)
    
    # 2. second_order_gradients
    # Compute second order gradient
    with tf.GradientTape() as tape2:
        tape2.watch(x)
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            y = nested_func(x)
        grad_x = tape1.gradient(y, x)
    grad2_x = tape2.gradient(grad_x, x)
    
    # Second derivative: d²/dx² [x^(3^n)] = (3^n)*(3^n - 1)*x^(3^n - 2)
    expected_grad2 = power * (power - 1) * (x ** (power - 2))
    assert_tensors_close(grad2_x, expected_grad2, rtol=1e-6, atol=1e-6)
    
    # 3. deep_nesting_stability
    # Test that deep nesting doesn't cause numerical issues
    if nesting_level >= 3:
        # For deep nesting, ensure gradients are finite
        assert tf.reduce_all(tf.math.is_finite(grad_x)), "Gradient should be finite for deep nesting"
        assert tf.reduce_all(tf.math.is_finite(grad2_x)), "Second gradient should be finite for deep nesting"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
@pytest.mark.parametrize("mode,dtype,shape,function_type,extreme_values", [
    ("eager", tf.float32, [2, 2], "log1pexp", True),
])
def test_numerical_stability_boundary_tests(mode, dtype, shape, function_type, extreme_values):
    """TC-05: 数值稳定性边界测试"""
    # Skip if mode not supported
    if mode == "graph" and tf.executing_eagerly():
        pytest.skip("Graph mode test requires non-eager execution")
    
    # Create test tensor with extreme values
    if extreme_values:
        # Create tensor with extreme values: very large positive, very large negative, zero
        if shape == [2, 2]:
            x_values = tf.constant([[100.0, -100.0], [0.0, 50.0]], dtype=dtype)
        else:
            # For other shapes, create a mix of extreme values
            x = tf.random.uniform(shape, minval=-100.0, maxval=100.0, dtype=dtype)
            # Set some elements to extreme values
            mask_large = tf.random.uniform(shape) > 0.7
            mask_small = tf.random.uniform(shape) > 0.7
            x = tf.where(mask_large, 100.0, x)
            x = tf.where(mask_small, -100.0, x)
            x_values = x
    else:
        x_values = create_test_tensor(shape, dtype)
    
    # Define the operation
    if function_type == "log1pexp":
        # Use the log1pexp function from header
        @tf.custom_gradient
        def decorated_func(x):
            return log1pexp(x)
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    
    # Weak assertions (epoch 1-4)
    # 1. output_finite
    y = decorated_func(x_values)
    assert tf.reduce_all(tf.math.is_finite(y)), "Output should be finite for all inputs"
    
    # 2. gradient_finite
    with tf.GradientTape() as tape:
        tape.watch(x_values)
        y = decorated_func(x_values)
    grad_x = tape.gradient(y, x_values)
    assert grad_x is not None, "Gradient should exist"
    assert tf.reduce_all(tf.math.is_finite(grad_x)), "Gradient should be finite for all inputs"
    
    # 3. no_nan_inf
    # Check that there are no NaN or Inf values
    assert not tf.reduce_any(tf.math.is_nan(y)), "Output should not contain NaN"
    assert not tf.reduce_any(tf.math.is_inf(y)), "Output should not contain Inf"
    assert not tf.reduce_any(tf.math.is_nan(grad_x)), "Gradient should not contain NaN"
    assert not tf.reduce_any(tf.math.is_inf(grad_x)), "Gradient should not contain Inf"
    
    # 4. numerical_stability
    # Compare with naive implementation for stability
    def naive_log1pexp(x):
        return tf.math.log(1 + tf.exp(x))
    
    # Compute naive gradient (may be unstable)
    with tf.GradientTape() as tape:
        tape.watch(x_values)
        y_naive = naive_log1pexp(x_values)
    grad_naive = tape.gradient(y_naive, x_values)
    
    # Check if naive implementation produces NaN/Inf
    naive_has_issues = tf.reduce_any(tf.math.is_nan(grad_naive)) or tf.reduce_any(tf.math.is_inf(grad_naive))
    
    # Custom gradient should be stable even if naive is not
    if naive_has_issues:
        print(f"Warning: Naive implementation has numerical issues at extreme values")
        print(f"Custom gradient remains stable: {tf.reduce_all(tf.math.is_finite(grad_x))}")
    
    # Strong assertions (epoch 5)
    # 1. gradient_precision_bound
    # For log1pexp, gradient should be between 0 and 1
    grad_lower_bound = 0.0
    grad_upper_bound = 1.0
    if dtype == tf.float32:
        tolerance = 1e-6
    else:
        tolerance = 1e-12
    
    # Check gradient bounds with tolerance
    assert tf.reduce_all(grad_x >= grad_lower_bound - tolerance), f"Gradient should be >= {grad_lower_bound}"
    assert tf.reduce_all(grad_x <= grad_upper_bound + tolerance), f"Gradient should be <= {grad_upper_bound}"
    
    # 2. extreme_value_handling
    # Test specific extreme values
    test_cases = [
        (100.0, 1.0),      # Large positive: gradient approaches 1
        (-100.0, 0.0),     # Large negative: gradient approaches 0
        (0.0, 0.5),        # Zero: gradient is 0.5
    ]
    
    for test_val, expected_grad in test_cases:
        x_test = tf.constant(test_val, dtype=dtype)
        with tf.GradientTape() as tape:
            tape.watch(x_test)
            y_test = decorated_func(x_test)
        grad_test = tape.gradient(y_test, x_test)
        
        # Allow some tolerance for extreme values
        if abs(test_val) > 50:
            tolerance = 1e-3
        else:
            tolerance = 1e-6
        
        assert abs(grad_test.numpy() - expected_grad) < tolerance, \
            f"At x={test_val}, expected gradient ~{expected_grad}, got {grad_test.numpy()}"
    
    # 3. overflow_underflow_protection
    # Test with even more extreme values
    extreme_test_values = tf.constant([1000.0, -1000.0, 1e6, -1e6], dtype=dtype)
    for val in extreme_test_values:
        x_extreme = tf.constant(val, dtype=dtype)
        with tf.GradientTape() as tape:
            tape.watch(x_extreme)
            y_extreme = decorated_func(x_extreme)
        grad_extreme = tape.gradient(y_extreme, x_extreme)
        
        # Should not overflow or underflow
        assert tf.math.is_finite(y_extreme), f"Output should be finite at x={val}"
        assert tf.math.is_finite(grad_extreme), f"Gradient should be finite at x={val}"
        
        # Gradient should still be in [0, 1] range
        assert 0.0 - 1e-3 <= grad_extreme.numpy() <= 1.0 + 1e-3, \
            f"Gradient at x={val} should be in [0,1], got {grad_extreme.numpy()}"
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:FOOTER START ====
# Additional test cases for error scenarios and edge cases

def test_invalid_function_return_type():
    """Test that function must return tuple (y, grad_fn)"""
    # Function that doesn't return a tuple
    def invalid_func(x):
        return x * x  # Missing grad_fn
    
    with pytest.raises((TypeError, ValueError)):
        @tf.custom_gradient
        def decorated_func(x):
            return invalid_func(x)
        
        # Try to use it
        x = tf.constant(2.0)
        decorated_func(x)

def test_grad_fn_wrong_signature():
    """Test that grad_fn must have correct signature"""
    # Function with grad_fn that has wrong signature
    def func_with_wrong_grad(x):
        def wrong_grad():  # Missing upstream parameter
            return x
        return x * x, wrong_grad
    
    with pytest.raises((TypeError, ValueError)):
        @tf.custom_gradient
        def decorated_func(x):
            return func_with_wrong_grad(x)
        
        x = tf.constant(2.0)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = decorated_func(x)
        tape.gradient(y, x)

def test_custom_gradient_decorator_factory():
    """Test decorator factory mode (f=None)"""
    # Create decorator factory
    custom_grad_decorator = tf.custom_gradient
    
    # Use it as decorator
    @custom_grad_decorator
    def my_func(x):
        def grad_fn(upstream):
            return upstream * 2 * x
        return x * x, grad_fn
    
    # Test the decorated function
    x = tf.constant(3.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = my_func(x)
    grad = tape.gradient(y, x)
    
    assert y.numpy() == 9.0
    assert grad.numpy() == 6.0

def test_stop_gradient_behavior():
    """Test interaction with tf.stop_gradient"""
    @tf.custom_gradient
    def func_with_stop_grad(x):
        # Use stop_gradient in forward pass
        x_stopped = tf.stop_gradient(x)
        def grad_fn(upstream):
            # Custom gradient that ignores original gradient
            return upstream * 0.5  # Always return half of upstream
        return x_stopped * x_stopped, grad_fn
    
    x = tf.constant(4.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = func_with_stop_grad(x)
    grad = tape.gradient(y, x)
    
    # With stop_gradient, the gradient should be 0.5 * upstream
    # For f(x) = x^2, normal gradient would be 2x = 8
    # With our custom grad_fn returning 0.5 * upstream, gradient should be 0.5 * 8 = 4
    # But wait, upstream is 1.0 for scalar output, so grad_fn returns 0.5
    # Actually, let's think: y = x_stopped^2, dy/dx_stopped = 2x_stopped = 8
    # But x_stopped = stop_gradient(x), so dy/dx = 0
    # Our grad_fn returns 0.5 * upstream, and upstream = 1.0, so final gradient = 0.5
    assert grad.numpy() == 0.5

if __name__ == "__main__":
    # Simple test runner for debugging
    import sys
    pytest.main([sys.argv[0], "-v"])
# ==== BLOCK:FOOTER END ====