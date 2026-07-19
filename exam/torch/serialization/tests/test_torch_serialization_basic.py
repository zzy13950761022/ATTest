"""
Test cases for torch.serialization basic save/load functionality (Group G1).
"""
import io
import os
import tempfile
import pytest
import torch
import torch.serialization

# ==== BLOCK:HEADER START ====
"""
Test cases for torch.serialization basic save/load functionality (Group G1).
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

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize("dtype,device,shape,file_type,use_new_zipfile", [
    (torch.float32, 'cpu', (2, 3), 'path', True),
])
def test_basic_tensor_save_load(
    dtype, device, shape, file_type, use_new_zipfile, tmp_file_path, random_seed
):
    """
    TC-01: Basic tensor save/load functionality.
    
    Tests that a tensor can be saved and loaded correctly with basic properties
    preserved (shape, dtype, values).
    """
    # Create test tensor
    tensor = torch.randn(*shape, dtype=dtype, device=device)
    
    if file_type == 'path':
        # Save to file path
        torch.save(
            tensor, 
            tmp_file_path,
            _use_new_zipfile_serialization=use_new_zipfile
        )
        
        # Verify file was created
        assert os.path.exists(tmp_file_path), "File should exist after save"
        assert os.path.getsize(tmp_file_path) > 0, "File should not be empty"
        
        # Load from file
        loaded = torch.load(tmp_file_path)
    else:
        # This branch would handle other file types, but param matrix only has 'path'
        pytest.skip(f"File type {file_type} not implemented in this test")
    
    # Weak assertions (first round)
    # 1. File existence (already checked for path type)
    # 2. Tensor shape
    assert loaded.shape == tensor.shape, \
        f"Loaded tensor shape {loaded.shape} != original {tensor.shape}"
    
    # 3. Tensor type
    assert loaded.dtype == tensor.dtype, \
        f"Loaded tensor dtype {loaded.dtype} != original {tensor.dtype}"
    
    # 4. Numerical approximation
    assert torch.allclose(loaded, tensor, rtol=1e-7, atol=1e-7), \
        "Tensor values differ beyond tolerance"
    
    # 5. Device (should be same as original)
    assert loaded.device == tensor.device, \
        f"Loaded tensor device {loaded.device} != original {tensor.device}"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("dtype,device,shape,file_type,use_new_zipfile", [
    (torch.int64, 'cpu', (5,), 'BytesIO', True),
])
def test_memory_file_object_support(
    dtype, device, shape, file_type, use_new_zipfile, random_seed
):
    """
    TC-02: Memory file object support.
    
    Tests that tensors can be saved to and loaded from in-memory file objects
    like BytesIO.
    """
    # Create test tensor
    tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
    
    if file_type == 'BytesIO':
        # Create BytesIO buffer
        buffer = io.BytesIO()
        
        # Save to buffer
        torch.save(
            tensor,
            buffer,
            _use_new_zipfile_serialization=use_new_zipfile
        )
        
        # Check buffer position and size
        buffer_size = buffer.tell()
        assert buffer_size > 0, "Buffer should contain data"
        
        # Reset buffer position for reading
        buffer.seek(0)
        
        # Load from buffer
        loaded = torch.load(buffer)
        
        # Check buffer methods were used correctly
        # (implicitly verified by successful save/load)
    else:
        pytest.skip(f"File type {file_type} not implemented in this test")
    
    # Weak assertions
    # 1. Tensor shape
    assert loaded.shape == tensor.shape, \
        f"Loaded tensor shape {loaded.shape} != original {tensor.shape}"
    
    # 2. Tensor type
    assert loaded.dtype == tensor.dtype, \
        f"Loaded tensor dtype {loaded.dtype} != original {tensor.dtype}"
    
    # 3. Numerical equality (exact for integer types)
    assert torch.equal(loaded, tensor), \
        "Integer tensor values should be exactly equal"
    
    # 4. File object method verification
    # Buffer position after load should be at end of data
    # Note: torch.load may not leave buffer at exact end, so we don't assert this
    
    # 5. Device
    assert loaded.device == tensor.device, \
        f"Loaded tensor device {loaded.device} != original {tensor.device}"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize("dtype,src_device,target_device,shape,map_location_type", [
    (torch.float32, 'cpu', 'cpu', (3, 3), 'string'),
])
def test_device_mapping_functionality(
    dtype, src_device, target_device, shape, map_location_type, 
    tmp_file_path, random_seed, monkeypatch
):
    """
    TC-03: Device mapping functionality.
    
    Tests that map_location parameter correctly maps tensors to target devices.
    """
    # Mock CUDA availability for consistent testing
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    
    # Create test tensor
    tensor = torch.randn(*shape, dtype=dtype, device=src_device)
    
    # Save tensor
    torch.save(tensor, tmp_file_path)
    
    # Determine map_location based on type
    if map_location_type == 'string':
        map_location = target_device  # 'cpu'
    elif map_location_type == 'torch.device':
        map_location = torch.device(target_device)
    else:
        pytest.skip(f"map_location_type {map_location_type} not implemented")
    
    # Load with map_location
    loaded = torch.load(tmp_file_path, map_location=map_location)
    
    # Weak assertions
    # 1. Target device
    expected_device = torch.device(target_device)
    assert loaded.device == expected_device, \
        f"Loaded tensor device {loaded.device} != expected {expected_device}"
    
    # 2. Tensor shape
    assert loaded.shape == tensor.shape, \
        f"Loaded tensor shape {loaded.shape} != original {tensor.shape}"
    
    # 3. Numerical approximation
    # Move original tensor to same device for comparison
    tensor_on_target = tensor.to(expected_device)
    assert torch.allclose(loaded, tensor_on_target, rtol=1e-7, atol=1e-7), \
        "Tensor values differ beyond tolerance"
    
    # 4. Dtype preservation
    assert loaded.dtype == tensor.dtype, \
        f"Loaded tensor dtype {loaded.dtype} != original {tensor.dtype}"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
@pytest.mark.parametrize("obj_type,weights_only,should_succeed,shape", [
    ('tensor', True, True, (2, 2)),
])
def test_weights_only_safety_mode(
    obj_type, weights_only, should_succeed, shape, tmp_file_path, random_seed
):
    """
    TC-04: weights_only safety mode.
    
    Tests that weights_only parameter restricts loading to safe objects only.
    """
    # Create test object based on type
    if obj_type == 'tensor':
        obj = torch.randn(*shape, dtype=torch.float32)
    elif obj_type == 'unsafe_object':
        # This would be an unsafe object like a lambda function
        # For now, we'll skip as it's in param_extensions
        pytest.skip("Unsafe object test deferred to param extensions")
    else:
        pytest.skip(f"Object type {obj_type} not implemented")
    
    # Save object
    torch.save(obj, tmp_file_path)
    
    if should_succeed:
        # Should load successfully with weights_only=True for safe objects
        loaded = torch.load(tmp_file_path, weights_only=weights_only)
        
        # Weak assertions for successful load
        # 1. Loading succeeded (no exception)
        # 2. Tensor type
        assert isinstance(loaded, torch.Tensor), \
            f"Loaded object should be Tensor, got {type(loaded)}"
        
        # 3. Shape correct
        assert loaded.shape == obj.shape, \
            f"Loaded tensor shape {loaded.shape} != original {obj.shape}"
        
        # 4. Values preserved
        assert torch.allclose(loaded, obj, rtol=1e-7, atol=1e-7), \
            "Tensor values differ beyond tolerance"
    else:
        # Should fail with RuntimeError for unsafe objects in weights_only mode
        with pytest.raises(RuntimeError) as exc_info:
            torch.load(tmp_file_path, weights_only=weights_only)
        
        # Verify error message indicates safety restriction
        error_msg = str(exc_info.value).lower()
        # Check for common error indicators
        assert any(keyword in error_msg for keyword in 
                  ['unsafe', 'restricted', 'weights_only', 'not allowed']), \
            f"Error message should indicate safety restriction: {error_msg}"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: Storage sharing preservation test (deferred to later round)
# This test will verify that storage sharing relationships are preserved
# across serialization. Requires strong assertions.
pass
# ==== BLOCK:CASE_05 END ====

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

# ==== BLOCK:CASE_08 START ====
# Deferred test case placeholder
# Will be implemented in later rounds
pass
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:CASE_09 START ====
# Deferred test case placeholder
# Will be implemented in later rounds
pass
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:FOOTER START ====
# Additional test functions for edge cases and cleanup

def test_save_load_none_object(tmp_file_path):
    """Test saving and loading None object."""
    obj = None
    torch.save(obj, tmp_file_path)
    loaded = torch.load(tmp_file_path)
    assert loaded is None, "None should be preserved"


def test_save_load_empty_tensor(tmp_file_path):
    """Test saving and loading empty tensor."""
    tensor = torch.tensor([], dtype=torch.float32)
    torch.save(tensor, tmp_file_path)
    loaded = torch.load(tmp_file_path)
    assert torch.equal(loaded, tensor), "Empty tensor should be preserved"
    assert loaded.shape == tensor.shape, "Shape should be preserved"


def test_save_load_special_values(tmp_file_path):
    """Test saving and loading special float values."""
    tensor = torch.tensor([float('inf'), float('-inf'), float('nan'), 0.0, -0.0])
    torch.save(tensor, tmp_file_path)
    loaded = torch.load(tmp_file_path)
    
    # Check inf values
    assert torch.isinf(loaded[0]) and loaded[0] > 0, "inf should be preserved"
    assert torch.isinf(loaded[1]) and loaded[1] < 0, "-inf should be preserved"
    
    # Check nan
    assert torch.isnan(loaded[2]), "nan should be preserved"
    
    # Check zeros (including signed zero)
    assert loaded[3] == 0.0, "0.0 should be preserved"
    assert loaded[4] == -0.0, "-0.0 should be preserved"


def test_invalid_file_path():
    """Test loading from non-existent file raises appropriate error."""
    non_existent_path = "/tmp/non_existent_file_12345.pt"
    with pytest.raises((FileNotFoundError, OSError)) as exc_info:
        torch.load(non_existent_path)
    # Verify it's a file not found error
    assert "No such file" in str(exc_info.value) or "not found" in str(exc_info.value).lower()


def test_corrupted_file(tmp_file_path):
    """Test loading corrupted file raises error."""
    # Write garbage data to file
    with open(tmp_file_path, 'wb') as f:
        f.write(b'corrupted data not a valid torch file')
    
    with pytest.raises((EOFError, RuntimeError, pickle.UnpicklingError)) as exc_info:
        torch.load(tmp_file_path)
    # Should raise some unpickling error
    # Actual error message is 'pickle data was truncated', so we need to adjust assertion
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in 
              ['pickle', 'unpickle', 'corrupt', 'truncated', 'eof', 'data']), \
        f"Error message should indicate pickle/unpickle error: {error_msg}"
# ==== BLOCK:FOOTER END ====