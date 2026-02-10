"""
Test cases for torch.serialization advanced functionality (Group G2).
"""
import io
import os
import pickle
import tempfile
import pytest
import torch
import torch.serialization

# ==== BLOCK:HEADER START ====
"""
Test cases for torch.serialization advanced functionality (Group G2).
"""
import io
import os
import pickle
import tempfile
import pytest
import torch
import torch.serialization


@pytest.fixture
def tmp_file_path():
    """Create a temporary file path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    return 42


def assert_tensors_equal(t1, t2, rtol=1e-7, atol=1e-7):
    """Helper to assert two tensors are equal with tolerance."""
    assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} != {t2.shape}"
    assert t1.dtype == t2.dtype, f"Dtype mismatch: {t1.dtype} != {t2.dtype}"
    assert torch.allclose(t1, t2, rtol=rtol, atol=atol), "Tensor values differ"
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: Device mapping functionality
# This test is already implemented in G1 file, but should be moved here
# for proper grouping. For now, we'll keep a placeholder.
pass
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_06 START ====
# Deferred test case placeholder
# Will be implemented in later rounds
pass
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# Deferred test case placeholder
# Will be implemented in later rounds
pass
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:FOOTER START ====
# Additional test functions for advanced functionality

def test_map_location_callable(tmp_file_path):
    """Test map_location as a callable function."""
    # Create tensor on CPU
    tensor = torch.randn(3, 3, dtype=torch.float32)
    torch.save(tensor, tmp_file_path)
    
    # Define a callable map_location that always maps to CPU
    def map_to_cpu(storage, location):
        return storage
    
    # Load with callable map_location
    loaded = torch.load(tmp_file_path, map_location=map_to_cpu)
    
    # Should still be on CPU
    assert loaded.device == torch.device('cpu')
    assert torch.allclose(loaded, tensor, rtol=1e-7, atol=1e-7)


def test_map_location_dict(tmp_file_path):
    """Test map_location as a dictionary."""
    # Create tensor on CPU
    tensor = torch.randn(2, 2, dtype=torch.float32)
    torch.save(tensor, tmp_file_path)
    
    # Map from CPU to CPU using dict
    map_location = {'cpu': 'cpu'}
    loaded = torch.load(tmp_file_path, map_location=map_location)
    
    # Should be on CPU
    assert loaded.device == torch.device('cpu')
    assert torch.allclose(loaded, tensor, rtol=1e-7, atol=1e-7)


def test_weights_only_with_unsafe_object(tmp_file_path):
    """Test weights_only=True with unsafe object should fail."""
    # Create an unsafe object (a lambda function)
    unsafe_obj = lambda x: x + 1
    
    # Save the unsafe object
    torch.save(unsafe_obj, tmp_file_path)
    
    # Try to load with weights_only=True - should fail
    with pytest.raises(RuntimeError) as exc_info:
        torch.load(tmp_file_path, weights_only=True)
    
    # Verify error message indicates safety restriction
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in 
              ['unsafe', 'restricted', 'weights_only', 'not allowed']), \
        f"Error message should indicate safety restriction: {error_msg}"


def test_file_like_object_with_seek(tmp_file_path):
    """Test loading from file-like object with seek/tell methods."""
    # Create tensor
    tensor = torch.randn(5, 5, dtype=torch.float32)
    
    # Save to file
    torch.save(tensor, tmp_file_path)
    
    # Open file in binary mode and load
    with open(tmp_file_path, 'rb') as f:
        loaded = torch.load(f)
    
    # Verify tensor
    assert torch.allclose(loaded, tensor, rtol=1e-7, atol=1e-7)
    assert loaded.device == tensor.device
    assert loaded.dtype == tensor.dtype
    assert loaded.shape == tensor.shape


def test_legacy_zipfile_format(tmp_file_path):
    """Test compatibility with legacy zipfile format."""
    tensor = torch.randn(3, 4, dtype=torch.float32)
    
    # Save with legacy format
    torch.save(tensor, tmp_file_path, _use_new_zipfile_serialization=False)
    
    # Load - should work with both formats
    loaded = torch.load(tmp_file_path)
    
    # Verify
    assert torch.allclose(loaded, tensor, rtol=1e-7, atol=1e-7)
    assert loaded.shape == tensor.shape
    assert loaded.dtype == tensor.dtype
# ==== BLOCK:FOOTER END ====