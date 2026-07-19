"""Test CUDA unavailable scenarios for torch.cuda.random module.

This file contains tests specifically for CUDA unavailable scenarios
that were identified as coverage gaps in the analysis plan.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock


class TestTorchCudaRandomCudaUnavailable:
    """Test CUDA unavailable scenarios for torch.cuda.random functions."""
    
    def test_get_rng_state_cuda_unavailable(self):
        """Test get_rng_state when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.get_rng_state()
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
    def test_get_rng_state_all_cuda_unavailable(self):
        """Test get_rng_state_all when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.get_rng_state_all()
            
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
    
    def test_set_rng_state_all_cuda_unavailable(self):
        """Test set_rng_state_all when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            test_states = [torch.ByteTensor(100).random_(0, 256) for _ in range(2)]
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.set_rng_state_all(test_states)
            
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
    
    def test_seed_cuda_unavailable(self):
        """Test seed when CUDA is not available.
        
        According to docstring: "It's safe to call this function if CUDA is not available;
        in that case, it is silently ignored."
        """
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise any exception
            torch.cuda.seed()
    
    def test_seed_all_cuda_unavailable(self):
        """Test seed_all when CUDA is not available.
        
        According to docstring: "It's safe to call this function if CUDA is not available;
        in that case, it is silently ignored."
        """
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise any exception
            torch.cuda.seed_all()
    
    def test_initial_seed_cuda_unavailable(self):
        """Test initial_seed when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.initial_seed()
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
    def test_empty_state_list_cuda_unavailable(self):
        """Test empty state list handling when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.set_rng_state_all([])
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["cuda", "initializ", "available"]), \
                f"Error message should mention CUDA availability, got: {error_msg}"
    
    def test_invalid_device_cuda_unavailable(self):
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


def test_cuda_unavailable_module_functions_exist():
    """Test that module functions exist even when CUDA is not available."""
    import torch.cuda.random as cuda_random
    
    # These functions should exist regardless of CUDA availability
    expected_functions = [
        'get_rng_state', 'get_rng_state_all',
        'set_rng_state', 'set_rng_state_all',
        'manual_seed', 'manual_seed_all',
        'seed', 'seed_all', 'initial_seed'
    ]
    
    for func_name in expected_functions:
        assert hasattr(cuda_random, func_name), \
            f"torch.cuda.random should have function {func_name} even when CUDA not available"
        func = getattr(cuda_random, func_name)
        assert callable(func), f"{func_name} should be callable even when CUDA not available"


@pytest.mark.parametrize("seed_value", [0, 42, -1, 2147483647, -2147483648])
def test_extreme_seeds_cuda_unavailable(seed_value):
    """Test extreme seed values when CUDA is not available."""
    with patch('torch.cuda.is_available', return_value=False):
        # manual_seed and manual_seed_all should be silently ignored
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        
        # seed and seed_all should be silently ignored
        torch.cuda.seed()
        torch.cuda.seed_all()
        
        # No exceptions should be raised for these functions
        # according to their docstrings