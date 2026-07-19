"""
Test cases for tensorflow.python.compiler.mlir.mlir module - G2 group.
G2: convert_function 函数族
"""
import math
import pytest
import tensorflow as tf
from unittest import mock
from tensorflow.python.compiler.mlir import mlir
from tensorflow.python.framework import errors_impl
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function


# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G2 group
@pytest.fixture
def mock_pywrap_mlir():
    """Mock the pywrap_mlir module to isolate tests from C++ implementation."""
    with mock.patch.object(mlir, 'pywrap_mlir') as mock_pywrap:
        # Setup default mock behavior
        mock_pywrap.import_function.return_value = "module {}\n"
        yield mock_pywrap


@pytest.fixture
def simple_add_function():
    """Create a simple add function for testing."""
    @tf.function
    def add(a, b):
        return a + b
    
    # Get concrete function with float32 inputs
    concrete_fn = add.get_concrete_function(
        tf.TensorSpec(shape=None, dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.float32)
    )
    return concrete_fn


@pytest.fixture
def complex_function():
    """Create a complex function for testing."""
    @tf.function
    def complex_math(x, y):
        # Multiple operations
        add_result = x + y
        mul_result = x * y
        sub_result = x - y
        return add_result, mul_result, sub_result
    
    concrete_fn = complex_math.get_concrete_function(
        tf.TensorSpec(shape=None, dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.float32)
    )
    return concrete_fn


@pytest.fixture
def control_flow_function():
    """Create a function with control flow for testing."""
    @tf.function
    def control_flow(x):
        # Simple if-else control flow
        if tf.reduce_sum(x) > 0:
            return x * 2
        else:
            return x * -1
    
    concrete_fn = control_flow.get_concrete_function(
        tf.TensorSpec(shape=[3], dtype=tf.float32)
    )
    return concrete_fn


@pytest.fixture
def invalid_function():
    """Create an invalid function for testing."""
    # Return None to simulate invalid function
    return None


class TestMLIRModuleG2:
    """Test class for G2 group: convert_function 函数族."""
    
    # ==== BLOCK:HEADER END ====
    
    # ==== BLOCK:CASE_03 START ====
    @pytest.mark.parametrize(
        "function_type,pass_pipeline,show_debug_info,expected_output",
        [
            (
                "simple_add",
                "tf-standard-pipeline",
                False,
                "module {\n  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {\n    %0 = \"tf.Add\"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>\n    return %0 : tensor<f32>\n  }\n}\n"
            ),
            (
                "complex_function",
                "tf-standard-pipeline",
                False,
                "module {\n  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {\n    %0 = \"tf.Add\"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>\n    %1 = \"tf.Mul\"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>\n    %2 = \"tf.Sub\"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>\n    return %0, %1, %2 : tensor<f32>, tensor<f32>, tensor<f32>\n  }\n}\n"
            ),
            (
                "control_flow",
                "tf-standard-pipeline",
                False,
                "module {\n  func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {\n    %0 = \"tf.Const\"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>\n    %1 = \"tf.Sum\"(%arg0) {keep_dims = false} : (tensor<3xf32>) -> tensor<f32>\n    %2 = \"tf.Greater\"(%1, %0) : (tensor<f32>, tensor<f32>) -> tensor<i1>\n    %3 = \"tf.If\"(%2) {\n      then_branch = @then_branch, else_branch = @else_branch\n    } : (tensor<i1>) -> tensor<3xf32>\n    return %3 : tensor<3xf32>\n  }\n  func.func @then_branch(%arg0: tensor<3xf32>) -> tensor<3xf32> {\n    %0 = \"tf.Const\"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>\n    %1 = \"tf.Mul\"(%arg0, %0) : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>\n    return %1 : tensor<3xf32>\n  }\n  func.func @else_branch(%arg0: tensor<3xf32>) -> tensor<3xf32> {\n    %0 = \"tf.Const\"() {value = dense<-1.000000e+00> : tensor<f32>} : () -> tensor<f32>\n    %1 = \"tf.Mul\"(%arg0, %0) : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>\n    return %1 : tensor<3xf32>\n  }\n}\n"
            ),
        ]
    )
    def test_convert_function_basic_conversion(
        self,
        mock_pywrap_mlir,
        simple_add_function,
        complex_function,
        control_flow_function,
        function_type,
        pass_pipeline,
        show_debug_info,
        expected_output
    ):
        """
        TC-03: convert_function 基本转换
        Test basic conversion of ConcreteFunction to MLIR text.
        """
        # Arrange - select appropriate function based on function_type
        if function_type == "simple_add":
            concrete_function = simple_add_function
        elif function_type == "complex_function":
            concrete_function = complex_function
        elif function_type == "control_flow":
            concrete_function = control_flow_function
        else:
            pytest.fail(f"Unknown function_type: {function_type}")
        
        mock_pywrap_mlir.import_function.return_value = expected_output
        
        # Act
        result = mlir.convert_function(
            concrete_function=concrete_function,
            pass_pipeline=pass_pipeline,
            show_debug_info=show_debug_info
        )
        
        # Assert (weak assertions)
        # 1. returns_string
        assert isinstance(result, str), "Result should be a string"
        
        # 2. contains_module
        assert "module" in result.lower(), "Result should contain 'module'"
        
        # 3. no_exception
        # No exception should be raised (implicitly verified by reaching this point)
        
        # Verify mock was called with correct parameters
        mock_pywrap_mlir.import_function.assert_called_once_with(
            concrete_function,
            pass_pipeline,
            show_debug_info
        )
        
        # Verify the returned value matches expected
        assert result == expected_output, "Result should match expected MLIR output"
        
        # Additional checks based on function type
        if function_type == "simple_add":
            assert "tf.Add" in result, "Should contain Add operation"
        elif function_type == "complex_function":
            assert "tf.Add" in result and "tf.Mul" in result and "tf.Sub" in result, \
                "Should contain Add, Mul, and Sub operations"
        elif function_type == "control_flow":
            assert "tf.If" in result, "Should contain If operation for control flow"
    # ==== BLOCK:CASE_03 END ====
    
    # ==== BLOCK:CASE_04 START ====
    @pytest.mark.parametrize(
        "function_type,pass_pipeline,show_debug_info,expected_output",
        [
            (
                "simple_add",
                "custom-pipeline",
                True,
                "module {\n  // Debug info: location information\n  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {\n    %0 = \"tf.Add\"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>\n    return %0 : tensor<f32>\n  }\n}\n"
            ),
        ]
    )
    def test_convert_function_parameter_validation(
        self,
        mock_pywrap_mlir,
        simple_add_function,
        function_type,
        pass_pipeline,
        show_debug_info,
        expected_output
    ):
        """
        TC-04: convert_function 参数验证
        Test parameter validation for convert_function.
        """
        # Arrange
        mock_pywrap_mlir.import_function.return_value = expected_output
        
        # Act
        result = mlir.convert_function(
            concrete_function=simple_add_function,
            pass_pipeline=pass_pipeline,
            show_debug_info=show_debug_info
        )
        
        # Assert (weak assertions)
        # 1. returns_string
        assert isinstance(result, str), "Result should be a string"
        
        # 2. no_exception
        # No exception should be raised (implicitly verified by reaching this point)
        
        # 3. debug_info_present (when show_debug_info=True)
        if show_debug_info:
            # Check for debug info indicators in the output
            assert any(
                debug_indicator in result.lower()
                for debug_indicator in ["debug", "location", "loc", "//"]
            ), "Debug info should be present when show_debug_info=True"
        
        # Verify mock was called with correct parameters
        mock_pywrap_mlir.import_function.assert_called_once_with(
            simple_add_function,
            pass_pipeline,
            show_debug_info
        )
        
        # Verify custom pipeline was used
        assert pass_pipeline == "custom-pipeline", "Custom pipeline should be used"
        
        # Verify the returned value
        assert result == expected_output, "Result should match expected MLIR output"
        
        # Verify function type specific content
        if function_type == "simple_add":
            assert "tf.Add" in result, "Should contain Add operation"
    # ==== BLOCK:CASE_04 END ====
    
    # ==== BLOCK:CASE_06 START ====
    @pytest.mark.parametrize(
        "function_type,pass_pipeline,show_debug_info,error_message",
        [
            (
                "invalid_function",
                "tf-standard-pipeline",
                False,
                "Invalid ConcreteFunction"
            ),
        ]
    )
    def test_convert_function_exception_handling(
        self,
        mock_pywrap_mlir,
        invalid_function,
        function_type,
        pass_pipeline,
        show_debug_info,
        error_message
    ):
        """
        TC-06: convert_function 异常处理
        Test exception handling for convert_function.
        """
        # Arrange
        mock_pywrap_mlir.import_function.side_effect = errors_impl.InvalidArgumentError(
            None, None, error_message
        )
        
        # Act & Assert
        with pytest.raises(errors_impl.InvalidArgumentError) as exc_info:
            mlir.convert_function(
                concrete_function=invalid_function,
                pass_pipeline=pass_pipeline,
                show_debug_info=show_debug_info
            )
        
        # Assert (weak assertions)
        # 1. raises_exception
        # Verified by pytest.raises context manager
        
        # 2. exception_type
        assert exc_info.type == errors_impl.InvalidArgumentError, \
            "Should raise InvalidArgumentError"
        
        # Verify mock was called with correct parameters
        mock_pywrap_mlir.import_function.assert_called_once_with(
            invalid_function,
            pass_pipeline,
            show_debug_info
        )
    # ==== BLOCK:CASE_06 END ====
    
    # ==== BLOCK:FOOTER START ====
    # Additional test cases and cleanup for G2 group
    def test_convert_function_edge_cases(self, mock_pywrap_mlir, simple_add_function):
        """Test edge cases for convert_function."""
        # Test with empty pass pipeline
        mock_pywrap_mlir.import_function.return_value = "module {}\n"
        
        result = mlir.convert_function(
            concrete_function=simple_add_function,
            pass_pipeline="",
            show_debug_info=False
        )
        
        assert isinstance(result, str)
        assert result == "module {}\n"
        
        # Test with very long pass pipeline
        long_pipeline = "tf-standard-pipeline," + ",".join([f"pass{i}" for i in range(10)])
        mock_pywrap_mlir.import_function.return_value = "module {}\n"
        
        result = mlir.convert_function(
            concrete_function=simple_add_function,
            pass_pipeline=long_pipeline,
            show_debug_info=False
        )
        
        assert isinstance(result, str)
        mock_pywrap_mlir.import_function.assert_called_with(
            simple_add_function,
            long_pipeline,
            False
        )
        
        # Test with different tensor specs
        @tf.function
        def add_int(a, b):
            return a + b
        
        int_concrete_fn = add_int.get_concrete_function(
            tf.TensorSpec(shape=[2, 3], dtype=tf.int32),
            tf.TensorSpec(shape=[2, 3], dtype=tf.int32)
        )
        
        mock_pywrap_mlir.import_function.return_value = "module {\n  func.func @main(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {\n    %0 = \"tf.Add\"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>\n    return %0 : tensor<2x3xi32>\n  }\n}\n"
        
        result = mlir.convert_function(
            concrete_function=int_concrete_fn,
            pass_pipeline="tf-standard-pipeline",
            show_debug_info=False
        )
        
        assert isinstance(result, str)
        assert "tensor<2x3xi32>" in result, "Should contain int32 tensor type"
    # ==== BLOCK:FOOTER END ====