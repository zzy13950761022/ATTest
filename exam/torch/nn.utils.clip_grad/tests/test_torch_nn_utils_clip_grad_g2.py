import math
import pytest
import torch
import warnings
from torch.nn.utils import clip_grad_norm_, clip_grad_value_, clip_grad_norm

# ==== BLOCK:HEADER START ====
# Test class and fixtures for G2 group
class TestClipGradValueAndDeprecated:
    """Test cases for clip_grad_value_ and deprecated clip_grad_norm functions"""
    
    @pytest.fixture
    def fixed_seed(self):
        """Fix random seed for reproducibility"""
        torch.manual_seed(42)
        return None
    
    def _create_gradients(self, shape, num_params, dtype=torch.float32, device='cpu'):
        """Helper to create gradients with random values"""
        params = []
        for i in range(num_params):
            p = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
            # Set gradient with values that may need clipping
            p.grad = torch.randn_like(p) * 2.0  # Scale to ensure some gradients need clipping
            params.append(p)
        return params
    
    def _get_grad_values(self, parameters):
        """Helper to get all gradient values as a flat tensor"""
        return torch.cat([p.grad.data.flatten() for p in parameters])
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
    @pytest.mark.parametrize("dtype,device,shape,num_params,clip_value", [
        (torch.float32, 'cpu', (2, 3), 3, 0.5),
    ])
    def test_clip_grad_value_basic(self, fixed_seed, dtype, device, shape, num_params, clip_value):
        """TC-03: clip_grad_value_ 基本功能"""
        # Create parameters with gradients
        parameters = self._create_gradients(shape, num_params, dtype, device)
        
        # Store original gradients for comparison
        original_grads = [p.grad.clone() for p in parameters]
        
        # Apply gradient value clipping
        result = clip_grad_value_(parameters, clip_value)
        
        # Weak assertions
        # 1. no_return_value: should return None
        assert result is None, "clip_grad_value_ should return None"
        
        # 2. gradients_clamped: all gradient values should be within [-clip_value, clip_value]
        for p in parameters:
            assert torch.all(p.grad >= -clip_value - 1e-6), f"Gradient values should be >= -{clip_value}"
            assert torch.all(p.grad <= clip_value + 1e-6), f"Gradient values should be <= {clip_value}"
        
        # 3. clamp_range_correct: values outside range should be clamped
        gradients_modified = False
        for i, p in enumerate(parameters):
            if not torch.allclose(p.grad, original_grads[i], rtol=1e-5):
                gradients_modified = True
                # Check that values were properly clamped
                original = original_grads[i]
                clipped = p.grad
                # Values > clip_value should be set to clip_value
                mask_gt = original > clip_value
                if mask_gt.any():
                    assert torch.allclose(clipped[mask_gt], torch.tensor(clip_value, dtype=dtype)), \
                        f"Values > {clip_value} should be clamped to {clip_value}"
                # Values < -clip_value should be set to -clip_value
                mask_lt = original < -clip_value
                if mask_lt.any():
                    assert torch.allclose(clipped[mask_lt], torch.tensor(-clip_value, dtype=dtype)), \
                        f"Values < -{clip_value} should be clamped to -{clip_value}"
                # Values within range should remain unchanged
                mask_within = (original >= -clip_value) & (original <= clip_value)
                if mask_within.any():
                    assert torch.allclose(clipped[mask_within], original[mask_within]), \
                        "Values within range should remain unchanged"
        
        # At least some gradients should be modified (since we scaled them)
        assert gradients_modified, "Some gradients should have been clipped"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
    def test_clip_grad_norm_deprecation_warning(self, fixed_seed):
        """TC-04: clip_grad_norm 弃用警告"""
        # Create a parameter with gradient
        tensor = torch.randn(2, 2, dtype=torch.float32, requires_grad=True)
        tensor.grad = torch.randn_like(tensor)
        
        # Store original gradient for comparison
        original_grad = tensor.grad.clone()
        
        # Mock warnings.warn to capture the warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Call the deprecated function
            total_norm = clip_grad_norm(tensor, max_norm=1.0, norm_type=2.0, error_if_nonfinite=False)
            
            # Weak assertions
            # 1. deprecation_warning: should issue a deprecation warning
            assert len(w) > 0, "Should issue a deprecation warning"
            assert any("deprecated" in str(warning.message).lower() for warning in w), \
                "Warning should indicate deprecation"
            
            # 2. function_works: function should still work
            assert total_norm is not None, "Function should return a value"
            assert isinstance(total_norm, torch.Tensor), "Should return a tensor"
            
            # 3. same_as_norm_: should behave the same as clip_grad_norm_
            # Reset gradient
            tensor.grad = original_grad.clone()
            # Call clip_grad_norm_ for comparison
            total_norm_ = clip_grad_norm_(tensor, max_norm=1.0, norm_type=2.0, error_if_nonfinite=False)
            
            # Both should return similar values (within tolerance)
            assert torch.allclose(total_norm, total_norm_, rtol=1e-5), \
                f"clip_grad_norm {total_norm} should equal clip_grad_norm_ {total_norm_}"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_07 START ====
    @pytest.mark.parametrize("dtype,device,shape,num_params,clip_value", [
        (torch.float64, 'cpu', (3, 3, 3), 2, 0.1),
        (torch.float32, 'cuda', (2, 3), 3, 0.5),
    ])
    def test_clip_grad_value_different_dtypes_devices(self, fixed_seed, dtype, device, shape, num_params, clip_value):
        """Test clip_grad_value_ with different data types and devices"""
        # Skip CUDA tests if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create parameters with gradients
        parameters = self._create_gradients(shape, num_params, dtype, device)
        
        # Store original gradients for comparison
        original_grads = [p.grad.clone() for p in parameters]
        
        # Apply gradient value clipping
        result = clip_grad_value_(parameters, clip_value)
        
        # Assertions
        # 1. Should return None
        assert result is None, "clip_grad_value_ should return None"
        
        # 2. All gradient values should be within [-clip_value, clip_value]
        for p in parameters:
            assert torch.all(p.grad >= -clip_value - 1e-6), f"Gradient values should be >= -{clip_value}"
            assert torch.all(p.grad <= clip_value + 1e-6), f"Gradient values should be <= {clip_value}"
        
        # 3. Check dtype preservation
        for p in parameters:
            assert p.grad.dtype == dtype, f"Gradient dtype should remain {dtype}, got {p.grad.dtype}"
        
        # 4. Check device preservation
        for p in parameters:
            assert p.grad.device.type == device, f"Gradient device should remain {device}, got {p.grad.device.type}"
        
        # 5. Verify clamping logic
        gradients_modified = False
        for i, p in enumerate(parameters):
            if not torch.allclose(p.grad, original_grads[i], rtol=1e-5):
                gradients_modified = True
                original = original_grads[i]
                clipped = p.grad
                
                # Values > clip_value should be set to clip_value
                mask_gt = original > clip_value
                if mask_gt.any():
                    assert torch.allclose(clipped[mask_gt], torch.tensor(clip_value, dtype=dtype, device=device)), \
                        f"Values > {clip_value} should be clamped to {clip_value}"
                
                # Values < -clip_value should be set to -clip_value
                mask_lt = original < -clip_value
                if mask_lt.any():
                    assert torch.allclose(clipped[mask_lt], torch.tensor(-clip_value, dtype=dtype, device=device)), \
                        f"Values < -{clip_value} should be clamped to -{clip_value}"
                
                # Values within range should remain unchanged
                mask_within = (original >= -clip_value) & (original <= clip_value)
                if mask_within.any():
                    assert torch.allclose(clipped[mask_within], original[mask_within]), \
                        "Values within range should remain unchanged"
        
        # At least some gradients should be modified
        assert gradients_modified, "Some gradients should have been clipped"
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# TC-08: (deferred - will be implemented in later iteration)
# This test case is deferred and will be implemented in a later iteration
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Additional test cases for edge scenarios in G2 group

def test_clip_grad_value_invalid_clip_value():
    """Test clip_grad_value_ with invalid clip_value"""
    # Test with clip_value = 0
    tensor = torch.randn(2, 3, requires_grad=True)
    tensor.grad = torch.randn_like(tensor)
    original_grad = tensor.grad.clone()
    
    # clip_grad_value_ should not raise an exception for clip_value = 0
    # It will clamp all gradients to 0
    result = clip_grad_value_(tensor, clip_value=0)
    assert result is None, "Should return None"
    # All gradients should be set to 0
    assert torch.all(tensor.grad == 0), "With clip_value=0, all gradients should be set to 0"
    
    # Test with clip_value = -1.0
    tensor2 = torch.randn(2, 3, requires_grad=True)
    tensor2.grad = torch.randn_like(tensor2)
    original_grad2 = tensor2.grad.clone()
    
    # clip_grad_value_ should not raise an exception for clip_value = -1.0
    # It will clamp with min=1.0, max=-1.0 (min > max)
    result2 = clip_grad_value_(tensor2, clip_value=-1.0)
    assert result2 is None, "Should return None"
    # With min=1.0, max=-1.0, all values should be clamped to max=-1.0
    # because clamp_(min, max) works even when min > max
    assert torch.all(tensor2.grad == -1.0), "With clip_value=-1.0, all gradients should be set to -1.0"

def test_clip_grad_value_single_tensor():
    """Test clip_grad_value_ with single tensor (not list)"""
    tensor = torch.randn(2, 3, requires_grad=True)
    tensor.grad = torch.randn_like(tensor) * 3.0  # Some values will be outside [-0.5, 0.5]
    
    original_grad = tensor.grad.clone()
    result = clip_grad_value_(tensor, clip_value=0.5)
    
    # Should return None
    assert result is None, "Should return None"
    
    # All values should be within [-0.5, 0.5]
    assert torch.all(tensor.grad >= -0.5 - 1e-6), "Values should be >= -0.5"
    assert torch.all(tensor.grad <= 0.5 + 1e-6), "Values should be <= 0.5"
    
    # Check that values were properly clamped
    mask_gt = original_grad > 0.5
    if mask_gt.any():
        assert torch.allclose(tensor.grad[mask_gt], torch.tensor(0.5)), \
            "Values > 0.5 should be clamped to 0.5"
    
    mask_lt = original_grad < -0.5
    if mask_lt.any():
        assert torch.allclose(tensor.grad[mask_lt], torch.tensor(-0.5)), \
            "Values < -0.5 should be clamped to -0.5"

def test_clip_grad_value_empty_parameters():
    """Test clip_grad_value_ with empty parameters list"""
    parameters = []
    result = clip_grad_value_(parameters, clip_value=0.5)
    assert result is None, "Empty parameters should return None"

def test_clip_grad_norm_deprecated_exact_message():
    """Test that clip_grad_norm shows the exact deprecation message"""
    tensor = torch.randn(2, 2, requires_grad=True)
    tensor.grad = torch.randn_like(tensor)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Call the deprecated function
        clip_grad_norm(tensor, max_norm=1.0)
        
        # Check warning message contains expected text
        assert len(w) == 1, "Should issue exactly one warning"
        warning_msg = str(w[0].message)
        # Check for common deprecation indicators
        assert any(keyword in warning_msg.lower() for keyword in ["deprecated", "clip_grad_norm_"]), \
            f"Warning message should mention deprecation or clip_grad_norm_. Got: {warning_msg}"
# ==== BLOCK:FOOTER END ====