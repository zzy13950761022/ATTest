import math
import pytest
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# ==== BLOCK:HEADER START ====
# Test file for torch.nn.utils.weight_norm - Group G2: Edge Cases and Error Handling
# 
# This file contains tests for edge cases and error handling of weight_norm:
# - Invalid module type error handling (CASE_06 - implemented)
# - Non-existent parameter name errors (CASE_07 - implemented in epoch 2)
# - Invalid dim parameter type errors (CASE_08 - deferred)
# - Conv2d layer weight normalization (CASE_09 - deferred)
# 
# Note: This is epoch 3/5, using weak assertions only.
# Deferred tests will be implemented in subsequent epochs.
# ==== BLOCK:HEADER END ====

class TestWeightNormG2:
    """Test cases for weight_norm edge cases and error handling (Group G2)."""
    
    def test_invalid_module_type_error(self):
        """TC-06: 无效模块类型错误处理
        
        Test that weight_norm raises AttributeError when given invalid module type.
        Weak assertions: raises_type_error (adjusted to AttributeError based on actual behavior), 
        error_message_contains
        Strong assertions: specific_error_type, error_message_matches
        
        Note: The function actually raises AttributeError when given non-module objects
        because it tries to access module._forward_pre_hooks attribute.
        """
        # Test with various invalid module types
        
        # Test 1: String instead of module
        with pytest.raises(AttributeError) as exc_info:
            weight_norm("not_a_module", name='weight', dim=0)
        
        # Weak assertion: raises_attribute_error (not TypeError)
        assert exc_info.type is AttributeError, \
            f"Should raise AttributeError for string input, got {exc_info.type}"
        
        # Weak assertion: error_message_contains - check error message mentions attribute
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ['attribute', 'has no attribute', '_forward_pre_hooks']), \
            f"Error message should mention missing attribute, got: {error_msg}"
        
        # STRONG ASSERTION 1: specific_error_type
        # Verify the exact error type is AttributeError (not TypeError or other)
        assert exc_info.type is AttributeError, \
            "Error type should be exactly AttributeError, not a subclass or other type"
        
        # STRONG ASSERTION 2: error_message_matches
        # Check for specific error message patterns
        error_msg = str(exc_info.value)
        
        # The error should mention the missing attribute '_forward_pre_hooks'
        # or indicate that the object doesn't have that attribute
        expected_patterns = [
            '_forward_pre_hooks',
            'has no attribute',
            "'str' object has no attribute",
            "object has no attribute"
        ]
        
        pattern_found = any(pattern in error_msg for pattern in expected_patterns)
        assert pattern_found, \
            f"Error message should match expected pattern. Got: {error_msg}"
        
        # Test 2: Integer instead of module
        with pytest.raises(AttributeError) as exc_info:
            weight_norm(123, name='weight', dim=0)
        
        assert exc_info.type is AttributeError, \
            f"Should raise AttributeError for integer input, got {exc_info.type}"
        
        # Verify error message for integer
        int_error_msg = str(exc_info.value)
        assert 'int' in int_error_msg or '123' in int_error_msg or 'object' in int_error_msg, \
            f"Error message for integer should mention int type. Got: {int_error_msg}"
        
        # Test 3: List instead of module
        with pytest.raises(AttributeError) as exc_info:
            weight_norm([1, 2, 3], name='weight', dim=0)
        
        assert exc_info.type is AttributeError, \
            f"Should raise AttributeError for list input, got {exc_info.type}"
        
        # Test 4: None instead of module
        with pytest.raises(AttributeError) as exc_info:
            weight_norm(None, name='weight', dim=0)
        
        assert exc_info.type is AttributeError, \
            f"Should raise AttributeError for None input, got {exc_info.type}"
        
        # Test 5: Dictionary instead of module
        with pytest.raises(AttributeError) as exc_info:
            weight_norm({'key': 'value'}, name='weight', dim=0)
        
        assert exc_info.type is AttributeError, \
            f"Should raise AttributeError for dict input, got {exc_info.type}"
        
        # Additional strong assertion: verify error consistency
        # All invalid types should raise AttributeError, not other exceptions
        invalid_inputs = ["string", 123, [1, 2, 3], None, {'key': 'value'}, 3.14, True]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(AttributeError) as exc_info:
                weight_norm(invalid_input, name='weight', dim=0)
            assert exc_info.type is AttributeError, \
                f"Input {type(invalid_input)} should raise AttributeError, got {exc_info.type}"
        
        # Additional check: valid module should not raise error
        valid_module = nn.Linear(10, 20)
        try:
            result = weight_norm(valid_module, name='weight', dim=0)
            assert result is valid_module, "Valid module should work"
            # Verify no AttributeError is raised for valid module
        except AttributeError as e:
            pytest.fail(f"Valid Linear module should not raise AttributeError: {e}")
    
    def test_nonexistent_parameter_name_error(self):
        """TC-07: 不存在的参数名称错误
        
        Test that weight_norm raises AttributeError when given non-existent parameter name.
        Weak assertions: raises_attribute_error, error_message_contains
        Strong assertions: specific_attribute_error, parameter_name_in_message
        
        Param matrix: Linear(10,20), name='nonexistent', dim=0, device='cpu', dtype='float32'
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create a Linear module with standard parameters
        module = nn.Linear(in_features=10, out_features=20)
        
        # Test with non-existent parameter name
        with pytest.raises(AttributeError) as exc_info:
            weight_norm(module, name='nonexistent', dim=0)
        
        # Weak assertion: raises_attribute_error
        assert exc_info.type is AttributeError, \
            f"Should raise AttributeError for non-existent parameter, got {exc_info.type}"
        
        # Weak assertion: error_message_contains - check error message mentions parameter name
        error_msg = str(exc_info.value).lower()
        
        # Check for various possible error message patterns
        error_indicators = ['parameter', 'nonexistent', 'attribute', 'has no attribute']
        found_indicator = any(indicator in error_msg for indicator in error_indicators)
        
        assert found_indicator, \
            f"Error message should mention parameter or attribute, got: {error_msg}"
        
        # STRONG ASSERTION 1: specific_attribute_error
        # Verify it's specifically an AttributeError about missing parameter
        assert exc_info.type is AttributeError, \
            "Error should be exactly AttributeError type"
        
        # Check error message contains specific information
        error_msg_full = str(exc_info.value)
        
        # The error should indicate that the parameter doesn't exist
        assert 'parameter' in error_msg_full.lower() or 'attribute' in error_msg_full.lower(), \
            f"Error should mention parameter or attribute. Got: {error_msg_full}"
        
        # STRONG ASSERTION 2: parameter_name_in_message
        # Verify the parameter name appears in the error message
        # The parameter name 'nonexistent' should be in the error message
        assert 'nonexistent' in error_msg_full.lower(), \
            f"Error message should contain parameter name 'nonexistent'. Got: {error_msg_full}"
        
        # Additional verification: check exact error message pattern
        # The error should mention that the module doesn't have the parameter
        expected_patterns = [
            'has no attribute',
            "'nonexistent'",
            'parameter',
            'Linear'
        ]
        
        patterns_found = sum(1 for pattern in expected_patterns if pattern.lower() in error_msg_full.lower())
        assert patterns_found >= 2, \
            f"Error message should contain at least 2 expected patterns. Got: {error_msg_full}"
        
        # Additional check: verify the module still has its original parameters
        assert hasattr(module, 'weight'), "Module should still have 'weight' parameter"
        assert hasattr(module, 'bias'), "Module should still have 'bias' parameter"
        
        # Verify that no new parameters were added
        assert not hasattr(module, 'nonexistent'), \
            "Module should not have 'nonexistent' parameter"
        assert not hasattr(module, 'nonexistent_g'), \
            "Module should not have 'nonexistent_g' parameter"
        assert not hasattr(module, 'nonexistent_v'), \
            "Module should not have 'nonexistent_v' parameter"
        
        # Test with another non-existent parameter name for consistency
        with pytest.raises(AttributeError) as exc_info2:
            weight_norm(module, name='invalid_param', dim=0)
        
        assert exc_info2.type is AttributeError, \
            f"Should raise AttributeError for 'invalid_param', got {exc_info2.type}"
        
        error_msg2 = str(exc_info2.value).lower()
        assert 'invalid_param' in error_msg2, \
            f"Error message should contain 'invalid_param'. Got: {error_msg2}"
        
        # Test with empty string as parameter name (edge case)
        with pytest.raises(AttributeError) as exc_info3:
            weight_norm(module, name='', dim=0)
        
        assert exc_info3.type is AttributeError, \
            f"Should raise AttributeError for empty parameter name, got {exc_info3.type}"
        
        # Test with special characters in parameter name
        with pytest.raises(AttributeError) as exc_info4:
            weight_norm(module, name='weight@special', dim=0)
        
        assert exc_info4.type is AttributeError, \
            f"Should raise AttributeError for special char parameter name, got {exc_info4.type}"
        
        # Positive control: test with valid parameter name should work
        try:
            result = weight_norm(nn.Linear(10, 20), name='weight', dim=0)
            assert result is not None, "Valid parameter name should work"
            assert hasattr(result, 'weight_g'), "Should have weight_g parameter"
            assert hasattr(result, 'weight_v'), "Should have weight_v parameter"
            
            # Verify no error for valid parameter
        except AttributeError as e:
            pytest.fail(f"Valid parameter name 'weight' should not raise AttributeError: {e}")
    
    # ==== BLOCK:CASE_08 START ====
    # TC-08: 无效dim参数类型错误 (DEFERRED - placeholder only)
    # Priority: Medium, Size: S, Max lines: 60
    # Param matrix: Linear(10,20), name='weight', dim='invalid', device='cpu', dtype='float32'
    # Weak asserts: raises_type_error, error_message_contains
    # ==== BLOCK:CASE_08 END ====
    
    # ==== BLOCK:CASE_09 START ====
    # TC-09: Conv2d层权重归一化 (DEFERRED - placeholder only)
    # Priority: Medium, Size: M, Max lines: 75
    # Param matrix: Conv2d(3,16,kernel_size=3), name='weight', dim=0, device='cpu', dtype='float32'
    # Weak asserts: returns_module, has_g_param, has_v_param, conv_weight_reconstructed
    # ==== BLOCK:CASE_09 END ====

# ==== BLOCK:FOOTER START ====
# Footer for test_torch_nn_utils_weight_norm_g2.py
# 
# Additional notes:
# - All tests use fixed random seed for reproducibility
# - Weak assertions are used in epoch 3
# - CASE_06 fixed in epoch 3 (changed TypeError to AttributeError)
# - CASE_07 implemented in epoch 2
# - CASE_08 and CASE_09 are still deferred
# - Medium priority tests are being implemented incrementally
# ==== BLOCK:FOOTER END ====