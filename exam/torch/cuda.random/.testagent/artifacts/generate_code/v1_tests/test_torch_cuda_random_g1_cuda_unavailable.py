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
        """Test get_rng_state when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.get_rng_state()
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
    def test_set_rng_state_cuda_unavailable(self):
        """Test set_rng_state when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            test_state = torch.ByteTensor(100).random_(0, 256)
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.set_rng_state(test_state)
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
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
        """Test initial_seed when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.initial_seed()
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
    def test_invalid_device_index_cuda_unavailable(self):
        """Test invalid device index when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Test with invalid device index
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.get_rng_state(device=-1)
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
            
            # Test with large invalid index
            with pytest.raises(RuntimeError) as exc_info2:
                torch.cuda.get_rng_state(device=999)
            
            error_msg2 = str(exc_info2.value).lower()
            assert any(keyword in error_msg2 for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg2}"
    
    def test_non_byte_tensor_state_cuda_unavailable(self):
        """Test non-ByteTensor state when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Create non-ByteTensor state
            float_state = torch.FloatTensor(100).normal_()
            
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.set_rng_state(float_state)
            
            error_msg = str(exc_info.value).lower()
            # Error could be about CUDA availability or state type
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available", "type", "byte"]), \
                f"Error message should mention CUDA availability or state type, got: {error_msg}"
    
    def test_single_device_state_management_cuda_unavailable(self):
        """Test single device state management when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Test all G1 functions in CUDA unavailable scenario
            
            # get_rng_state should raise RuntimeError
            with pytest.raises(RuntimeError):
                torch.cuda.get_rng_state()
            
            # set_rng_state should raise RuntimeError
            test_state = torch.ByteTensor(100).random_(0, 256)
            with pytest.raises(RuntimeError):
                torch.cuda.set_rng_state(test_state)
            
            # manual_seed should be silently ignored
            torch.cuda.manual_seed(42)
            
            # seed should be silently ignored
            torch.cuda.seed()
            
            # initial_seed should raise RuntimeError
            with pytest.raises(RuntimeError):
                torch.cuda.initial_seed()
            
            # Test with device parameter
            with pytest.raises(RuntimeError):
                torch.cuda.get_rng_state(device=0)
            
            with pytest.raises(RuntimeError):
                torch.cuda.set_rng_state(test_state, device=0)
    
    def test_device_string_format_cuda_unavailable(self):
        """Test device string format when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            test_state = torch.ByteTensor(100).random_(0, 256)
            
            # Test with string device
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.get_rng_state(device="cuda")
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
            
            with pytest.raises(RuntimeError) as exc_info2:
                torch.cuda.set_rng_state(test_state, device="cuda")
            
            error_msg2 = str(exc_info2.value).lower()
            assert any(keyword in error_msg2 for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg2}"
    
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