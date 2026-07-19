"""Test CUDA unavailable scenarios for G2 (multi-device) functions.

This file supplements the G2 tests with specific CUDA unavailable scenarios
that were identified as coverage gaps in the analysis plan.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock


class TestTorchCudaRandomG2CudaUnavailable:
    """Test CUDA unavailable scenarios for G2 multi-device functions."""
    
    def test_get_rng_state_all_cuda_unavailable(self):
        """Test get_rng_state_all when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.get_rng_state_all()
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
    def test_set_rng_state_all_cuda_unavailable(self):
        """Test set_rng_state_all when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            test_states = [torch.ByteTensor(100).random_(0, 256) for _ in range(2)]
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.set_rng_state_all(test_states)
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
    def test_manual_seed_all_cuda_unavailable(self):
        """Test manual_seed_all when CUDA is not available.
        
        According to docstring: "It's safe to call this function if CUDA is not available;
        in that case, it is silently ignored."
        """
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise any exception
            torch.cuda.manual_seed_all(42)
            torch.cuda.manual_seed_all(0)
            torch.cuda.manual_seed_all(-1)
            torch.cuda.manual_seed_all(2147483647)
    
    def test_seed_all_cuda_unavailable(self):
        """Test seed_all when CUDA is not available.
        
        According to docstring: "It's safe to call this function if CUDA is not available;
        in that case, it is silently ignored."
        """
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise any exception
            torch.cuda.seed_all()
    
    def test_empty_state_list_cuda_unavailable(self):
        """Test empty state list handling when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.set_rng_state_all([])
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
    def test_invalid_device_index_cuda_unavailable(self):
        """Test invalid device index when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Test with invalid device index for get_rng_state
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.get_rng_state(device=-1)
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
            
            # Test with invalid device index for set_rng_state
            valid_state = torch.ByteTensor(100).random_(0, 256)
            with pytest.raises(RuntimeError) as exc_info2:
                torch.cuda.set_rng_state(valid_state, device=-1)
            
            error_msg2 = str(exc_info2.value).lower()
            assert any(keyword in error_msg2 for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg2}"
    
    def test_multi_device_batch_management_cuda_unavailable(self):
        """Test multi-device batch management when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Test all G2 functions in CUDA unavailable scenario
            
            # get_rng_state_all should raise RuntimeError
            with pytest.raises(RuntimeError):
                torch.cuda.get_rng_state_all()
            
            # set_rng_state_all should raise RuntimeError
            test_states = [torch.ByteTensor(100).random_(0, 256) for _ in range(2)]
            with pytest.raises(RuntimeError):
                torch.cuda.set_rng_state_all(test_states)
            
            # manual_seed_all should be silently ignored
            torch.cuda.manual_seed_all(42)
            
            # seed_all should be silently ignored
            torch.cuda.seed_all()
            
            # Test with mocked device_count
            with patch('torch.cuda.device_count', return_value=2):
                # Even with mocked device_count, CUDA unavailable should still raise
                with pytest.raises(RuntimeError):
                    torch.cuda.get_rng_state_all()
                
                with pytest.raises(RuntimeError):
                    torch.cuda.set_rng_state_all(test_states)
    
    def test_state_list_length_mismatch_cuda_unavailable(self):
        """Test state list length mismatch when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Create state list with wrong length
            wrong_length_states = [
                torch.ByteTensor(100).random_(0, 256),
                torch.ByteTensor(100).random_(0, 256),
                torch.ByteTensor(100).random_(0, 256)  # Three states
            ]
            
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.set_rng_state_all(wrong_length_states)
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
    def test_zero_length_tensor_state_cuda_unavailable(self):
        """Test zero-length tensor state when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            zero_tensor_states = [torch.ByteTensor(0), torch.ByteTensor(0)]
            
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.set_rng_state_all(zero_tensor_states)
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
    def test_none_state_list_cuda_unavailable(self):
        """Test None state list when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Test with None input
            try:
                torch.cuda.set_rng_state_all(None)
            except (RuntimeError, TypeError) as e:
                # Either exception type is acceptable
                pass
    
    @pytest.mark.parametrize("seed_value", [0, 42, -1, 2147483647, -2147483648])
    def test_extreme_seeds_cuda_unavailable(self, seed_value):
        """Test extreme seed values when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # manual_seed_all should be silently ignored for all seed values
            torch.cuda.manual_seed_all(seed_value)
            
            # seed_all should be silently ignored
            torch.cuda.seed_all()
            
            # No exceptions should be raised


def test_g2_functions_exist_cuda_unavailable():
    """Test that G2 functions exist even when CUDA is not available."""
    import torch.cuda.random as cuda_random
    
    g2_functions = [
        'get_rng_state_all',
        'set_rng_state_all', 
        'manual_seed_all',
        'seed_all'
    ]
    
    for func_name in g2_functions:
        assert hasattr(cuda_random, func_name), \
            f"torch.cuda.random should have function {func_name} even when CUDA not available"
        func = getattr(cuda_random, func_name)
        assert callable(func), f"{func_name} should be callable even when CUDA not available"