"""
Test cases for tensorflow.python.eager.def_function (G2 group).
"""
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.eager import def_function

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==== BLOCK:HEADER START ====
# Test class for def_function tests (G2 group)
class TestDefFunctionG2:
    """Test cases for tensorflow.python.eager.def_function (G2 group)."""
    
    def setup_method(self):
        """Setup method for each test."""
        # Clear any existing function caches
        tf.config.run_functions_eagerly(False)
        
    def teardown_method(self):
        """Teardown method for each test."""
        # Ensure we're not in eager mode for subsequent tests
        tf.config.run_functions_eagerly(False)
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
    def test_variable_creation_and_state_preservation(self):
        """CASE_03: 变量创建与状态保持 - 测试变量创建和状态保持"""
        import unittest.mock as mock
        
        # Test stateful counter function
        @def_function.function
        def stateful_counter():
            # Variable should be created only on first call
            if not hasattr(stateful_counter, 'counter'):
                stateful_counter.counter = tf.Variable(0, dtype=tf.int32)
            stateful_counter.counter.assign_add(1)
            return stateful_counter.counter.read_value()
        
        # Weak assertions
        # 1. variable_created_once: Variable should be created on first call
        # Mock Variable creation to verify it happens only once
        with mock.patch('tensorflow.Variable') as mock_var:
            # First call - should create variable
            result1 = stateful_counter()
            assert mock_var.call_count == 1, "Variable should be created on first call"
            
            # Reset mock to track subsequent calls
            mock_var.reset_mock()
            
            # Second call - should not create new variable
            result2 = stateful_counter()
            assert mock_var.call_count == 0, "Variable should not be recreated on subsequent calls"
        
        # Clean up and test actual behavior
        # Reset the counter attribute
        if hasattr(stateful_counter, 'counter'):
            delattr(stateful_counter, 'counter')
        
        # 2. state_preserved: State should be preserved between calls
        # Create a fresh function to test state preservation
        @def_function.function
        def fresh_counter():
            if not hasattr(fresh_counter, 'counter_var'):
                fresh_counter.counter_var = tf.Variable(0, dtype=tf.int32)
            fresh_counter.counter_var.assign_add(1)
            return fresh_counter.counter_var.read_value()
        
        # First call
        result1 = fresh_counter()
        assert result1.numpy() == 1, f"First call should return 1, got {result1.numpy()}"
        
        # Second call - should increment
        result2 = fresh_counter()
        assert result2.numpy() == 2, f"Second call should return 2, got {result2.numpy()}"
        
        # Third call - should increment again
        result3 = fresh_counter()
        assert result3.numpy() == 3, f"Third call should return 3, got {result3.numpy()}"
        
        # 3. callable: Verify function is callable
        assert callable(fresh_counter), "Function should be callable"
        
        # 4. basic_execution: Test basic execution works
        # Create another function with different initial value
        @def_function.function
        def counter_with_initial(initial_value):
            if not hasattr(counter_with_initial, 'counter'):
                counter_with_initial.counter = tf.Variable(initial_value, dtype=tf.int32)
            counter_with_initial.counter.assign_add(1)
            return counter_with_initial.counter.read_value()
        
        # Test with different initial values
        result_a = counter_with_initial(10)
        assert result_a.numpy() == 11, f"Should return 11, got {result_a.numpy()}"
        
        # Reset and test again
        if hasattr(counter_with_initial, 'counter'):
            delattr(counter_with_initial, 'counter')
        
        result_b = counter_with_initial(20)
        assert result_b.numpy() == 21, f"Should return 21, got {result_b.numpy()}"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
    def test_control_flow_statements_support(self):
        """CASE_04: 控制流语句支持 - 测试控制流语句支持"""
        # Test conditional flow function
        @def_function.function
        def conditional_flow(x):
            # Data-dependent control flow
            if tf.reduce_sum(x) > 0:
                return x * x  # Square if sum > 0
            else:
                return -x // 2  # Negative half if sum <= 0
        
        # Weak assertions
        # 1. if_statement_works: Test if statement with positive sum
        x_positive = tf.constant([1.0, 2.0, 3.0])
        result_positive = conditional_flow(x_positive)
        expected_positive = tf.constant([1.0, 4.0, 9.0])
        
        # Check that if branch was taken (squaring)
        np.testing.assert_allclose(result_positive.numpy(), expected_positive.numpy(), rtol=1e-5)
        
        # 2. if_statement_works: Test if statement with negative sum
        x_negative = tf.constant([-1.0, -2.0, -3.0])
        result_negative = conditional_flow(x_negative)
        # For negative sum, should return -x // 2
        # Note: // operator with floats does floor division
        expected_negative = tf.constant([0.0, 1.0, 1.0])  # -(-1)//2=0, -(-2)//2=1, -(-3)//2=1
        
        np.testing.assert_allclose(result_negative.numpy(), expected_negative.numpy(), rtol=1e-5)
        
        # 3. loop_execution: Test loop execution
        @def_function.function
        def loop_example(n):
            # Simple for loop
            total = tf.constant(0, dtype=tf.int32)
            for i in tf.range(n):
                total += i
            return total
        
        # Test loop with different values
        result_loop_5 = loop_example(5)
        assert result_loop_5.numpy() == 10, f"Sum 0..4 should be 10, got {result_loop_5.numpy()}"
        
        result_loop_10 = loop_example(10)
        assert result_loop_10.numpy() == 45, f"Sum 0..9 should be 45, got {result_loop_10.numpy()}"
        
        # 4. data_dependent_flow: Test more complex data-dependent flow
        @def_function.function
        def complex_flow(x, threshold):
            result = tf.constant(0.0)
            for val in x:
                if val > threshold:
                    result += val
                else:
                    result -= val
            return result
        
        x_data = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        threshold = tf.constant(3.0)
        result_complex = complex_flow(x_data, threshold)
        
        # Values > 3: 4.0 + 5.0 = 9.0
        # Values <= 3: -1.0 - 2.0 - 3.0 = -6.0
        # Total: 9.0 - 6.0 = 3.0
        expected_complex = tf.constant(3.0)
        np.testing.assert_allclose(result_complex.numpy(), expected_complex.numpy(), rtol=1e-5)
        
        # 5. basic_execution: Test basic execution with edge cases
        # Test with zero
        x_zero = tf.constant([0.0])
        result_zero = conditional_flow(x_zero)
        # Sum is 0, so should take else branch: -0 // 2 = 0
        expected_zero = tf.constant([0.0])
        np.testing.assert_allclose(result_zero.numpy(), expected_zero.numpy(), rtol=1e-5)
        
        # Test with single element
        x_single = tf.constant([5.0])
        result_single = conditional_flow(x_single)
        expected_single = tf.constant([25.0])
        np.testing.assert_allclose(result_single.numpy(), expected_single.numpy(), rtol=1e-5)
        
        # Verify functions are callable
        assert callable(conditional_flow), "conditional_flow should be callable"
        assert callable(loop_example), "loop_example should be callable"
        assert callable(complex_flow), "complex_flow should be callable"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_07 START ====
# Placeholder for CASE_07: (DEFERRED)
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# Placeholder for CASE_08: (DEFERRED)
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====