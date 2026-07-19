import torch
import pytest
import numpy as np
from typing import Callable, Tuple, Union
import math

# ==== BLOCK:HEADER START ====
# Test file for torch.autograd.gradcheck - Basic functionality tests
# Group: G1 - Basic function verification
# Target: torch.autograd.gradcheck
# ==== BLOCK:HEADER END ====

# Helper functions for test cases
def simple_polynomial(x: torch.Tensor) -> torch.Tensor:
    """Simple polynomial function: f(x) = x^3 + 2x^2 + 3x + 4"""
    return x.pow(3) + 2 * x.pow(2) + 3 * x + 4

def complex_function(z: torch.Tensor) -> torch.Tensor:
    """Complex-differentiable function: f(z) = z^2 + 2*z"""
    return z * z + 2 * z

def failing_function(x: torch.Tensor) -> torch.Tensor:
    """Function that fails gradient check due to incorrect implementation"""
    # This function has incorrect gradient implementation
    # Using a non-linear operation without proper gradient implementation
    # This will cause gradient check to fail
    return x * 2 + x.detach() * 0.1  # detach() breaks gradient flow

# Test class for basic gradcheck functionality
class TestGradcheckBasic:
    """Test cases for basic gradcheck functionality (Group G1)"""
    
    # ==== BLOCK:CASE_01 START ====
    @pytest.mark.parametrize("dtype,shape,eps,atol,rtol", [
        (torch.float64, (3, 3), 1e-6, 1e-5, 0.001),  # Base case from param_matrix
        (torch.float32, (5, 5), 1e-5, 1e-4, 0.01),   # Extension: single precision
    ])
    def test_basic_real_function_gradient_verification(self, dtype, shape, eps, atol, rtol):
        """
        TC-01: Basic real function gradient verification
        Test gradcheck with simple polynomial function on real-valued tensors.
        
        Weak assertions:
        - returns_bool: gradcheck returns boolean
        - no_exception: no exception raised during execution
        - basic_gradient_check: gradient check passes for correct function
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create input tensor with requires_grad=True
        x = torch.randn(*shape, dtype=dtype, requires_grad=True)
        
        # Test with simple polynomial function
        result = torch.autograd.gradcheck(
            func=simple_polynomial,
            inputs=(x,),
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=True,
            check_backward_ad=True
        )
        
        # Weak assertion: returns boolean
        assert isinstance(result, bool), f"gradcheck should return bool, got {type(result)}"
        
        # Weak assertion: no exception raised (implied by successful execution)
        # Weak assertion: gradient check should pass for correct function
        assert result is True, f"gradcheck should return True for correct polynomial function. dtype={dtype}, shape={shape}"
    # ==== BLOCK:CASE_01 END ====
    
    # ==== BLOCK:CASE_02 START ====
    @pytest.mark.parametrize("dtype,shape,eps,atol,rtol", [
        (torch.complex64, (2, 2), 1e-6, 1e-5, 0.001),   # Base case from param_matrix
        (torch.complex128, (3, 3), 1e-6, 1e-5, 0.001),  # Extension: double precision complex
    ])
    def test_complex_function_wirtinger_derivative_check(self, dtype, shape, eps, atol, rtol):
        """
        TC-02: Complex function Wirtinger derivative check
        Test gradcheck with complex-valued function to verify Wirtinger derivatives.
        
        Weak assertions:
        - returns_bool: gradcheck returns boolean
        - no_exception: no exception raised during execution
        - complex_gradient_check: gradient check passes for complex function
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create complex input tensor with requires_grad=True
        real_part = torch.randn(*shape, dtype=torch.float32 if dtype == torch.complex64 else torch.float64)
        imag_part = torch.randn(*shape, dtype=torch.float32 if dtype == torch.complex64 else torch.float64)
        z = torch.complex(real_part, imag_part)
        z.requires_grad_(True)
        
        # Test with complex function
        result = torch.autograd.gradcheck(
            func=complex_function,
            inputs=(z,),
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=True,
            check_backward_ad=True
        )
        
        # Weak assertion: returns boolean
        assert isinstance(result, bool), f"gradcheck should return bool, got {type(result)}"
        
        # Weak assertion: no exception raised (implied by successful execution)
        # Weak assertion: gradient check should pass for correct complex function
        assert result is True, f"gradcheck should return True for correct complex function. dtype={dtype}, shape={shape}"
    # ==== BLOCK:CASE_02 END ====
    
    # ==== BLOCK:CASE_04 START ====
    @pytest.mark.parametrize("dtype,shape,eps,atol,rtol", [
        (torch.float64, (2, 2), 1e-6, 1e-5, 0.001),  # Base case from param_matrix
    ])
    def test_forward_mode_automatic_differentiation_verification(self, dtype, shape, eps, atol, rtol):
        """
        TC-04: Forward mode automatic differentiation verification
        Test gradcheck with forward mode AD (check_forward_ad=True).
        
        Weak assertions:
        - returns_bool: gradcheck returns boolean
        - no_exception: no exception raised during execution
        - forward_ad_check: gradient check passes with forward mode AD
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create input tensor with requires_grad=True
        x = torch.randn(*shape, dtype=dtype, requires_grad=True)
        
        # Test with forward mode AD enabled
        result = torch.autograd.gradcheck(
            func=simple_polynomial,
            inputs=(x,),
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=True,
            check_forward_ad=True,  # Enable forward mode AD check
            check_backward_ad=False,  # Disable backward mode AD as specified
        )
        
        # Weak assertion: returns boolean
        assert isinstance(result, bool), f"gradcheck should return bool, got {type(result)}"
        
        # Weak assertion: no exception raised (implied by successful execution)
        # Weak assertion: gradient check should pass with forward mode AD
        assert result is True, f"gradcheck should return True with forward mode AD. dtype={dtype}, shape={shape}"
        
        # Additional test: compare forward and backward mode results
        # Test with both forward and backward mode AD
        result_both = torch.autograd.gradcheck(
            func=simple_polynomial,
            inputs=(x,),
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=True,
            check_forward_ad=True,
            check_backward_ad=True,  # Enable both modes
        )
        
        assert result_both is True, f"gradcheck should work with both forward and backward AD. dtype={dtype}, shape={shape}"
        
        # Test with only backward mode AD (default behavior)
        result_backward = torch.autograd.gradcheck(
            func=simple_polynomial,
            inputs=(x,),
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=True,
            check_forward_ad=False,  # Disable forward mode
            check_backward_ad=True,  # Enable backward mode
        )
        
        assert result_backward is True, f"gradcheck should work with backward mode AD only. dtype={dtype}, shape={shape}"
    # ==== BLOCK:CASE_04 END ====

# ==== BLOCK:FOOTER START ====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====