import io
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

# ==== BLOCK:HEADER START ====
import io
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from pathlib import Path


# Helper functions
def create_simple_script_module():
    """Create a simple ScriptModule for testing."""
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            return self.relu(self.linear(x))
    
    module = SimpleModule()
    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    for param in module.parameters():
        param.data.normal_(mean=0.0, std=1.0)
    
    return torch.jit.script(module)


def create_nested_script_module():
    """Create a nested ScriptModule for testing."""
    class SubModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, 3)
            
        def forward(self, x):
            return self.conv(x)
    
    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = SubModule()
            self.pool = nn.MaxPool2d(2)
            
        def forward(self, x):
            return self.pool(self.sub(x))
    
    module = NestedModule()
    torch.manual_seed(42)
    for param in module.parameters():
        param.data.normal_(mean=0.0, std=0.1)
    
    return torch.jit.script(module)


def compare_modules(module1, module2, rtol=1e-5, atol=1e-8):
    """Compare two ScriptModules for equality."""
    # Compare module structure
    assert type(module1) == type(module2)
    
    # Compare parameters
    params1 = dict(module1.named_parameters())
    params2 = dict(module2.named_parameters())
    
    assert set(params1.keys()) == set(params2.keys())
    
    for name in params1:
        p1 = params1[name]
        p2 = params2[name]
        assert torch.allclose(p1, p2, rtol=rtol, atol=atol), f"Parameter {name} mismatch"
    
    # Compare buffers if any
    buffers1 = dict(module1.named_buffers())
    buffers2 = dict(module2.named_buffers())
    
    if buffers1 or buffers2:
        assert set(buffers1.keys()) == set(buffers2.keys())
        for name in buffers1:
            b1 = buffers1[name]
            b2 = buffers2[name]
            assert torch.allclose(b1, b2, rtol=rtol, atol=atol), f"Buffer {name} mismatch"
    
    return True


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_module():
    """Fixture providing a simple ScriptModule."""
    return create_simple_script_module()


@pytest.fixture
def nested_module():
    """Fixture providing a nested ScriptModule."""
    return create_nested_script_module()


@pytest.fixture
def temp_file_path(temp_dir):
    """Fixture providing a temporary file path."""
    return os.path.join(temp_dir, "test_module.pt")
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize("module_type,file_type,device,extra_files", [
    ("simple_script_module", "path", "cpu", None),
])
def test_basic_file_serialization(module_type, file_type, device, extra_files, simple_module, temp_file_path):
    """
    TC-01: 基本文件序列化
    测试 torch.jit.save 和 torch.jit.load 的基本功能。
    """
    # Setup
    if module_type == "simple_script_module":
        module = simple_module
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    
    # Save module
    if file_type == "path":
        torch.jit.save(module, temp_file_path, _extra_files=extra_files)
        
        # Weak assertion: file_created
        assert os.path.exists(temp_file_path), "File should be created"
        assert os.path.getsize(temp_file_path) > 0, "File should not be empty"
    else:
        raise ValueError(f"Unknown file type: {file_type}")
    
    # Load module
    loaded_module = torch.jit.load(temp_file_path, map_location=device, _extra_files=extra_files)
    
    # Weak assertions
    # module_loaded
    assert loaded_module is not None, "Module should be loaded"
    assert isinstance(loaded_module, torch.jit.ScriptModule), "Loaded object should be ScriptModule"
    
    # structure_preserved
    # Compare module types and parameter names
    params_original = dict(module.named_parameters())
    params_loaded = dict(loaded_module.named_parameters())
    assert set(params_original.keys()) == set(params_loaded.keys()), "Parameter structure should be preserved"
    
    # parameters_equal (with tolerance for floating point)
    for name in params_original:
        p_orig = params_original[name]
        p_load = params_loaded[name]
        assert torch.allclose(p_orig, p_load, rtol=1e-5, atol=1e-8), f"Parameter {name} values should match"
    
    # Test forward pass consistency
    torch.manual_seed(123)
    test_input = torch.randn(2, 10)
    
    with torch.no_grad():
        output_original = module(test_input)
        output_loaded = loaded_module(test_input)
    
    assert torch.allclose(output_original, output_loaded, rtol=1e-5, atol=1e-8), "Forward pass outputs should match"
    
    # Clean up
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("module_type,file_type,device,extra_files", [
    ("simple_script_module", "buffer", "cpu", None),
])
def test_buffer_serialization(module_type, file_type, device, extra_files, simple_module):
    """
    TC-02: 缓冲区序列化
    测试使用内存缓冲区进行序列化和反序列化。
    """
    # Setup
    if module_type == "simple_script_module":
        module = simple_module
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    
    # Save to buffer
    buffer = io.BytesIO()
    torch.jit.save(module, buffer, _extra_files=extra_files)
    
    # Weak assertion: buffer_written
    buffer_size = buffer.tell()
    assert buffer_size > 0, "Buffer should contain data"
    
    # Weak assertion: buffer_readable
    buffer.seek(0)  # Reset buffer position
    buffer_content = buffer.read()
    assert len(buffer_content) == buffer_size, "Buffer content should be readable"
    
    # Load from buffer
    buffer.seek(0)  # Reset for loading
    loaded_module = torch.jit.load(buffer, map_location=device, _extra_files=extra_files)
    
    # Weak assertions
    # module_loaded
    assert loaded_module is not None, "Module should be loaded from buffer"
    assert isinstance(loaded_module, torch.jit.ScriptModule), "Loaded object should be ScriptModule"
    
    # structure_preserved
    params_original = dict(module.named_parameters())
    params_loaded = dict(loaded_module.named_parameters())
    assert set(params_original.keys()) == set(params_loaded.keys()), "Parameter structure should be preserved"
    
    # parameters_equal
    for name in params_original:
        p_orig = params_original[name]
        p_load = params_loaded[name]
        assert torch.allclose(p_orig, p_load, rtol=1e-5, atol=1e-8), f"Parameter {name} values should match"
    
    # Test forward pass
    torch.manual_seed(456)
    test_input = torch.randn(3, 10)
    
    with torch.no_grad():
        output_original = module(test_input)
        output_loaded = loaded_module(test_input)
    
    assert torch.allclose(output_original, output_loaded, rtol=1e-5, atol=1e-8), "Forward pass outputs should match"
    
    # Test that buffer can be reused
    buffer.seek(0)
    buffer2 = io.BytesIO(buffer.read())
    loaded_module2 = torch.jit.load(buffer2, map_location=device)
    params_loaded2 = dict(loaded_module2.named_parameters())
    
    for name in params_original:
        p_orig = params_original[name]
        p_load2 = params_loaded2[name]
        assert torch.allclose(p_orig, p_load2, rtol=1e-5, atol=1e-8), f"Parameter {name} should match in second load"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize("module_type,file_type,device,map_location,extra_files", [
    ("simple_script_module", "path", "cpu", "cpu", None),
])
def test_device_mapping(module_type, file_type, device, map_location, extra_files, simple_module, temp_file_path):
    """
    TC-03: 设备映射测试
    测试 torch.jit.load 的 map_location 参数。
    """
    # Setup
    if module_type == "simple_script_module":
        module = simple_module
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    
    # Ensure module is on CPU (should already be)
    module = module.cpu()
    
    # Save module
    torch.jit.save(module, temp_file_path, _extra_files=extra_files)
    
    # Load with map_location
    loaded_module = torch.jit.load(temp_file_path, map_location=map_location, _extra_files=extra_files)
    
    # Weak assertions
    # module_loaded
    assert loaded_module is not None, "Module should be loaded"
    assert isinstance(loaded_module, torch.jit.ScriptModule), "Loaded object should be ScriptModule"
    
    # device_correct
    # Check that all parameters are on the correct device
    expected_device = torch.device(map_location if isinstance(map_location, str) else map_location)
    
    for name, param in loaded_module.named_parameters():
        assert param.device == expected_device, f"Parameter {name} should be on {expected_device}, got {param.device}"
    
    # structure_preserved
    params_original = dict(module.named_parameters())
    params_loaded = dict(loaded_module.named_parameters())
    
    assert set(params_original.keys()) == set(params_loaded.keys()), "Parameter structure should be preserved"
    
    # parameters_equal (values should match regardless of device)
    for name in params_original:
        p_orig = params_original[name].cpu()  # Ensure CPU for comparison
        p_load = params_loaded[name].cpu()    # Ensure CPU for comparison
        assert torch.allclose(p_orig, p_load, rtol=1e-5, atol=1e-8), f"Parameter {name} values should match"
    
    # Test forward pass
    torch.manual_seed(789)
    test_input = torch.randn(4, 10)
    
    # Move input to same device as module if needed
    if map_location != "cpu":
        test_input = test_input.to(map_location)
    
    with torch.no_grad():
        output_original = module(test_input.cpu())  # Original module is on CPU
        output_loaded = loaded_module(test_input)
    
    # Move outputs to CPU for comparison
    output_original = output_original.cpu()
    output_loaded = output_loaded.cpu()
    
    assert torch.allclose(output_original, output_loaded, rtol=1e-5, atol=1e-8), "Forward pass outputs should match"
    
    # Test that map_location=None defaults to original device (CPU)
    loaded_module_default = torch.jit.load(temp_file_path, map_location=None, _extra_files=extra_files)
    for name, param in loaded_module_default.named_parameters():
        # According to documentation, modules are first loaded onto CPU
        assert param.device.type == "cpu", f"With map_location=None, parameter {name} should be on CPU"
    
    # Clean up
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
@pytest.mark.parametrize("module_type,file_type,device,extra_files", [
    ("simple_script_module", "file_object", "cpu", None),
])
def test_file_object_interface(module_type, file_type, device, extra_files, simple_module):
    """
    TC-04: 文件对象接口
    测试使用文件对象（而非路径）进行序列化和反序列化。
    """
    # Setup
    if module_type == "simple_script_module":
        module = simple_module
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    
    # Test 1: Save to file-like object with write method
    # Note: torch.jit.save documentation says file-like object needs write and flush,
    # but actual implementation may not call flush. We'll test both scenarios.
    mock_file = Mock(spec=['write', 'flush'])
    buffer_content = None
    
    # 使用side_effect来捕获写入的数据，同时保持Mock对象的属性
    def mock_write_side_effect(data):
        nonlocal buffer_content
        buffer_content = data
        return len(data) if data else 0
    
    mock_file.write.side_effect = mock_write_side_effect
    mock_file.flush.return_value = None
    
    # Save to mock file object
    torch.jit.save(module, mock_file, _extra_files=extra_files)
    
    # Weak assertion: file_methods_called
    assert mock_file.write.called, "write method should be called"
    # Note: flush may or may not be called depending on implementation
    # We'll log it but not assert it
    if mock_file.flush.called:
        print("flush method was called (implementation-dependent)")
    
    # Test 2: Load from file-like object with read/readline/tell/seek methods
    # Create a real BytesIO buffer to test loading
    real_buffer = io.BytesIO()
    torch.jit.save(module, real_buffer, _extra_files=extra_files)
    
    # Reset buffer for loading
    real_buffer.seek(0)
    
    # Create a mock that wraps the real buffer but verifies method calls
    class VerifyingFileObject:
        def __init__(self, buffer):
            self.buffer = buffer
            self.read_called = False
            self.readline_called = False
            self.tell_called = False
            self.seek_called = False
        
        def read(self, size=-1):
            self.read_called = True
            return self.buffer.read(size)
        
        def readline(self, size=-1):
            self.readline_called = True
            return self.buffer.readline(size)
        
        def tell(self):
            self.tell_called = True
            return self.buffer.tell()
        
        def seek(self, offset, whence=0):
            self.seek_called = True
            return self.buffer.seek(offset, whence)
    
    verifying_file = VerifyingFileObject(real_buffer)
    
    # Load from verifying file object
    loaded_module = torch.jit.load(verifying_file, map_location=device, _extra_files=extra_files)
    
    # Weak assertions for file methods
    assert verifying_file.read_called or verifying_file.readline_called, \
        "read or readline should be called during load"
    assert verifying_file.seek_called, "seek should be called during load"
    # tell may or may not be called depending on implementation
    
    # Weak assertion: module_loaded
    assert loaded_module is not None, "Module should be loaded from file object"
    assert isinstance(loaded_module, torch.jit.ScriptModule), "Loaded object should be ScriptModule"
    
    # structure_preserved
    params_original = dict(module.named_parameters())
    params_loaded = dict(loaded_module.named_parameters())
    assert set(params_original.keys()) == set(params_loaded.keys()), "Parameter structure should be preserved"
    
    # Test 3: Real file object (BytesIO) round trip
    buffer = io.BytesIO()
    torch.jit.save(module, buffer, _extra_files=extra_files)
    
    buffer.seek(0)
    loaded_module2 = torch.jit.load(buffer, map_location=device, _extra_files=extra_files)
    
    # Verify parameters match
    for name in params_original:
        p_orig = params_original[name]
        p_load = dict(loaded_module2.named_parameters())[name]
        assert torch.allclose(p_orig, p_load, rtol=1e-5, atol=1e-8), f"Parameter {name} should match"
    
    # Test forward pass consistency
    torch.manual_seed(999)
    test_input = torch.randn(5, 10)
    
    with torch.no_grad():
        output_original = module(test_input)
        output_loaded = loaded_module2(test_input)
    
    assert torch.allclose(output_original, output_loaded, rtol=1e-5, atol=1e-8), "Forward pass outputs should match"
    
    # Test 4: Invalid file object (missing required methods)
    invalid_file = Mock(spec=['read'])  # Missing seek method
    
    with pytest.raises((AttributeError, TypeError)):
        torch.jit.load(invalid_file, map_location=device)
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
@pytest.mark.parametrize("module_type,file_type,device,extra_files", [
    ("nested_script_module", "path", "cpu", None),
])
def test_complex_module_serialization(module_type, file_type, device, extra_files, nested_module, temp_dir):
    """
    TC-05: 复杂模块序列化
    测试嵌套ScriptModule的序列化和反序列化。
    """
    # Setup
    if module_type == "nested_script_module":
        module = nested_module
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    
    # Create a temporary file path
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', dir=temp_dir, delete=False) as tmp:
        file_path = tmp.name
    
    try:
        # Weak assertion: file_created
        # Save the module
        torch.jit.save(module, file_path, _extra_files=extra_files)
        assert os.path.exists(file_path), "File should be created after save"
        file_size = os.path.getsize(file_path)
        assert file_size > 0, "Saved file should not be empty"
        
        # Weak assertion: module_loaded
        # Load the module
        loaded_module = torch.jit.load(file_path, map_location=device, _extra_files=extra_files)
        assert loaded_module is not None, "Module should be loaded from file"
        assert isinstance(loaded_module, torch.jit.ScriptModule), "Loaded object should be ScriptModule"
        
        # Weak assertion: nested_structure_preserved
        # Check that the module structure is preserved
        # Get original module structure
        original_params = dict(module.named_parameters())
        loaded_params = dict(loaded_module.named_parameters())
        
        # Check parameter names match
        assert set(original_params.keys()) == set(loaded_params.keys()), \
            "Parameter structure should be preserved in nested module"
        
        # Check parameter values match
        for name in original_params:
            p_orig = original_params[name]
            p_load = loaded_params[name]
            assert torch.allclose(p_orig, p_load, rtol=1e-5, atol=1e-8), \
                f"Parameter {name} should match in nested module"
        
        # Check that submodules exist
        # Original module has 'sub' and 'pool' submodules
        assert hasattr(module, 'sub'), "Original module should have 'sub' submodule"
        assert hasattr(module, 'pool'), "Original module should have 'pool' submodule"
        
        # Loaded module should also have these submodules
        assert hasattr(loaded_module, 'sub'), "Loaded module should have 'sub' submodule"
        assert hasattr(loaded_module, 'pool'), "Loaded module should have 'pool' submodule"
        
        # Check that submodules are ScriptModules
        assert isinstance(module.sub, torch.jit.ScriptModule), \
            "Original submodule should be ScriptModule"
        assert isinstance(loaded_module.sub, torch.jit.ScriptModule), \
            "Loaded submodule should be ScriptModule"
        
        # Test forward pass consistency
        torch.manual_seed(999)
        
        # Create appropriate input for the nested module (Conv2d expects 4D input)
        if module_type == "nested_script_module":
            # Nested module expects 4D input: (batch, channels, height, width)
            test_input = torch.randn(2, 3, 32, 32)
        else:
            test_input = torch.randn(5, 10)
        
        with torch.no_grad():
            output_original = module(test_input)
            output_loaded = loaded_module(test_input)
        
        assert torch.allclose(output_original, output_loaded, rtol=1e-5, atol=1e-8), \
            "Forward pass outputs should match for nested module"
        
        # Additional check: verify the hierarchy is correct
        # The nested module should have a Conv2d in its 'sub' submodule
        assert hasattr(module.sub, 'conv'), "sub.submodule should have 'conv' attribute"
        assert isinstance(module.sub.conv, torch.nn.Module), "conv should be a Module"
        
        # Same for loaded module
        assert hasattr(loaded_module.sub, 'conv'), "Loaded sub.submodule should have 'conv' attribute"
        assert isinstance(loaded_module.sub.conv, torch.nn.Module), "Loaded conv should be a Module"
        
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# TC-06: 额外文件参数测试 (DEFERRED - will be implemented in later iteration)
# This test case is deferred and will be implemented in a later iteration.
# It will test the _extra_files parameter for saving and loading additional files.
# Parameters: TBD based on param_extensions in test_plan.json
pass
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# TC-07: GPU设备映射测试 (DEFERRED - will be implemented in later iteration)
# This test case is deferred and will be implemented in a later iteration.
# It will test device mapping to GPU devices (requires CUDA availability).
# Parameters: TBD based on param_extensions in test_plan.json
pass
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# TC-08: 异常场景测试 (DEFERRED - will be implemented in later iteration)
# This test case is deferred and will be implemented in a later iteration.
# It will test error handling for invalid inputs and edge cases.
# Parameters: TBD based on requirements.md error scenarios
pass
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:CASE_09 START ====
# TC-09: 边界条件测试 (DEFERRED - will be implemented in later iteration)
# This test case is deferred and will be implemented in a later iteration.
# It will test boundary conditions like empty modules, large tensors, etc.
# Parameters: TBD based on requirements.md boundary values
pass
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:FOOTER START ====
# Additional utility tests that don't fit into the main test cases

def test_save_invalid_module():
    """Test that saving a non-ScriptModule raises an error."""
    class RegularModule(nn.Module):
        def forward(self, x):
            return x * 2
    
    regular_module = RegularModule()
    
    with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
        # torch.jit.save will call regular_module.save() which doesn't exist
        # This raises AttributeError, not TypeError or RuntimeError
        with pytest.raises(AttributeError):
            torch.jit.save(regular_module, tmp.name)


def test_load_nonexistent_file():
    """Test that loading a non-existent file raises an error."""
    non_existent_path = "/tmp/nonexistent_file_12345.pt"
    
    # Ensure the file doesn't exist
    if os.path.exists(non_existent_path):
        os.remove(non_existent_path)
    
    # torch.jit.load raises ValueError for non-existent files, not FileNotFoundError
    with pytest.raises(ValueError, match="does not exist"):
        torch.jit.load(non_existent_path)


def test_extra_files_empty_dict():
    """Test that empty _extra_files dictionary works correctly."""
    module = create_simple_script_module()
    
    with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
        # Save with empty extra files
        torch.jit.save(module, tmp.name, _extra_files={})
        
        # Load with empty extra files
        extra_files = {}
        loaded_module = torch.jit.load(tmp.name, _extra_files=extra_files)
        
        assert loaded_module is not None
        assert isinstance(loaded_module, torch.jit.ScriptModule)
        assert extra_files == {}  # Should remain empty


def test_module_save_method_equivalence():
    """Test that torch.jit.save is equivalent to module.save()."""
    module = create_simple_script_module()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "test1.pt")
        path2 = os.path.join(tmpdir, "test2.pt")
        
        # Save using torch.jit.save
        torch.jit.save(module, path1)
        
        # Save using module.save method
        module.save(path2)
        
        # Both files should exist and have similar sizes
        assert os.path.exists(path1)
        assert os.path.exists(path2)
        
        size1 = os.path.getsize(path1)
        size2 = os.path.getsize(path2)
        # Sizes might not be exactly equal due to metadata, but should be close
        assert abs(size1 - size2) < 100  # Allow small differences
        
        # Both should load correctly
        module1 = torch.jit.load(path1)
        module2 = torch.jit.load(path2)
        
        assert isinstance(module1, torch.jit.ScriptModule)
        assert isinstance(module2, torch.jit.ScriptModule)


if __name__ == "__main__":
    # Simple test runner for debugging
    import sys
# ==== BLOCK:FOOTER END ====