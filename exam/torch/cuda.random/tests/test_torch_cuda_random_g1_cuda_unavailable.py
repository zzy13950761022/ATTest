"""Test CUDA unavailable scenarios for G1 (single-device) functions.

This file supplements the G1 tests with specific CUDA unavailable scenarios
that were identified as coverage gaps in the analysis plan.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock


class TestTorchCudaRandomG1CudaUnavailable:
    """Test CUDA unavailable scenarios for G1 single-device functions."""
    
    def test_get_rng_state_cuda_unavailable(self):
        """Test get_rng_state when CUDA is not available.
        
        This function calls _lazy_init() which raises AssertionError
        when torch is not compiled with CUDA enabled.
        """
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(AssertionError) as exc_info:
                torch.cuda.get_rng_state()
            
            error_msg = str(exc_info.value).lower()
            assert "torch not compiled with cuda enabled" in error_msg, \
                f"Error message should be 'Torch not compiled with CUDA enabled', got: {error_msg}"
    
    def test_set_rng_state_cuda_unavailable(self):
        """Test set_rng_state when CUDA is not available.
        
        This function uses _lazy_call() which queues the call instead of
        executing immediately, so no exception should be raised.
        """
        with patch('torch.cuda.is_available', return_value=False):
            test_state = torch.ByteTensor(100).random_(0, 256)
            # Should not raise any exception
            torch.cuda.set_rng_state(test_state)
    
    def test_manual_seed_cuda_unavailable(self):
        """Test manual_seed when CUDA is not available.
        
        According to docstring: "It's safe to call this function if CUDA is not available;
        in that case, it is silently ignored."
        """
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise any exception
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed(-1)
            torch.cuda.manual_seed(2147483647)
    
    def test_seed_cuda_unavailable(self):
        """Test seed when CUDA is not available.
        
        According to docstring: "It's safe to call this function if CUDA is not available;
        in that case, it is silently ignored."
        """
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise any exception
            torch.cuda.seed()
    
    def test_initial_seed_cuda_unavailable(self):
        """Test initial_seed when CUDA is not available.
        
        This function calls _lazy_init() which raises AssertionError
        when torch is not compiled with CUDA enabled.
        """
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(AssertionError) as exc_info:
                torch.cuda.initial_seed()
            
            error_msg = str(exc_info.value).lower()
            assert "torch not compiled with cuda enabled" in error_msg, \
                f"Error message should be 'Torch not compiled with CUDA enabled', got: {error_msg}"
    
    def test_invalid_device_index_cuda_unavailable(self):
        """Test invalid device index when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Test with invalid device index for get_rng_state
            with pytest.raises(AssertionError) as exc_info:
                torch.cuda.get_rng_state(device=-1)
            
            error_msg = str(exc_info.value).lower()
            assert "torch not compiled with cuda enabled" in error_msg, \
                f"Error message should be 'Torch not compiled with CUDA enabled', got: {error_msg}"
            
            # Test with large invalid index for get_rng_state
            with pytest.raises(AssertionError) as exc_info2:
                torch.cuda.get_rng_state(device=999)
            
            error_msg2 = str(exc_info2.value).lower()
            assert "torch not compiled with cuda enabled" in error_msg2, \
                f"Error message should be 'Torch not compiled with CUDA enabled', got: {error_msg2}"
            
            # Test with invalid device index for set_rng_state
            # This should raise RuntimeError because torch.device('cuda', -1) fails immediately
            valid_state = torch.ByteTensor(100).random_(0, 256)
            with pytest.raises(RuntimeError) as exc_info3:
                torch.cuda.set_rng_state(valid_state, device=-1)
            
            error_msg3 = str(exc_info3.value).lower()
            assert "device" in error_msg3 or "index" in error_msg3 or "negative" in error_msg3, \
                f"Error message should mention device/index, got: {error_msg3}"
    
    def test_non_byte_tensor_state_cuda_unavailable(self):
        """Test non-ByteTensor state when CUDA is not available.
        
        set_rng_state with non-ByteTensor should not raise exception
        when CUDA is not available (call is queued via _lazy_call).
        """
        with patch('torch.cuda.is_available', return_value=False):
            # Create non-ByteTensor state
            float_state = torch.FloatTensor(100).normal_()
            
            # Should not raise any exception
            torch.cuda.set_rng_state(float_state)
    
    def test_single_device_state_management_cuda_unavailable(self):
        """Test single device state management when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Test all G1 functions in CUDA unavailable scenario
            
            # get_rng_state should raise AssertionError
            with pytest.raises(AssertionError):
                torch.cuda.get_rng_state()
            
            # set_rng_state should NOT raise exception for valid device (uses _lazy_call)
            test_state = torch.ByteTensor(100).random_(0, 256)
            torch.cuda.set_rng_state(test_state)  # No exception expected for default device
            
            # manual_seed should be silently ignored
            torch.cuda.manual_seed(42)
            
            # seed should be silently ignored
            torch.cuda.seed()
            
            # initial_seed should raise AssertionError
            with pytest.raises(AssertionError):
                torch.cuda.initial_seed()
            
            # Test with device parameter
            with pytest.raises(AssertionError):
                torch.cuda.get_rng_state(device=0)
            
            torch.cuda.set_rng_state(test_state, device=0)  # No exception expected for valid device index
    
    def test_device_string_format_cuda_unavailable(self):
        """Test device string format when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            test_state = torch.ByteTensor(100).random_(0, 256)
            
            # Test with string device
            with pytest.raises(AssertionError) as exc_info:
                torch.cuda.get_rng_state(device="cuda")
            
            error_msg = str(exc_info.value).lower()
            assert "torch not compiled with cuda enabled" in error_msg, \
                f"Error message should be 'Torch not compiled with CUDA enabled', got: {error_msg}"
            
            # set_rng_state should NOT raise exception (uses _lazy_call)
            torch.cuda.set_rng_state(test_state, device="cuda")  # No exception expected
    
    @pytest.mark.parametrize("seed_value", [0, 42, -1, 2147483647, -2147483648])
    def test_extreme_seeds_cuda_unavailable(self, seed_value):
        """Test extreme seed values when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # manual_seed should be silently ignored for all seed values
            torch.cuda.manual_seed(seed_value)
            
            # seed should be silently ignored
            torch.cuda.seed()
            
            # No exceptions should be raised


def test_g1_functions_exist_cuda_unavailable():
    """Test that G1 functions exist even when CUDA is not available."""
    import torch.cuda.random as cuda_random
    
    g1_functions = [
        'get_rng_state',
        'set_rng_state', 
        'manual_seed',
        'seed',
        'initial_seed'
    ]
    
    for func_name in g1_functions:
        assert hasattr(cuda_random, func_name), \
            f"torch.cuda.random should have function {func_name} even when CUDA not available"
        func = getattr(cuda_random, func_name)
        assert callable(func), f"{func_name} should be callable even when CUDA not available"