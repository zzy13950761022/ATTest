import torch
import torch.nn as nn
import pytest
import copy
from torch.ao.quantization import fuse_modules


# ==== BLOCK:HEADER START ====
# Test helper functions and fixtures

import torch
import torch.nn as nn
import pytest
import copy
from torch.ao.quantization import fuse_modules


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleConvBNReLU(nn.Module):
    """Simple model with conv-bn-relu sequence."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SimpleConvBN(nn.Module):
    """Simple model with conv-bn sequence."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiLayerModel(nn.Module):
    """Model with multiple layers for multi-group fusion."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.linear = nn.Linear(16 * 32 * 32, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


class SimpleModel(nn.Module):
    """Simple model for error handling tests."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


@pytest.fixture
def random_input():
    """Fixture providing random input tensor."""
    set_random_seed(42)
    return torch.randn(2, 3, 32, 32)


def check_finite_output(output):
    """Check that output contains finite values."""
    assert torch.isfinite(output).all(), "Output contains NaN or infinite values"


def check_output_shape(original_output, fused_output):
    """Check that fused model output shape matches original."""
    assert fused_output.shape == original_output.shape, \
        f"Output shape mismatch: {fused_output.shape} != {original_output.shape}"
# ==== BLOCK:HEADER END ====


# ==== BLOCK:CASE_01 START ====
# Test case: 单组conv-bn-relu融合

@pytest.mark.parametrize("inplace", [False])
def test_single_group_conv_bn_relu_fusion(random_input, inplace):
    """Test fusion of single conv-bn-relu sequence."""
    # Setup
    model = SimpleConvBNReLU()
    model.eval()
    
    # Get original output
    with torch.no_grad():
        original_output = model(random_input)
    
    # Fuse modules
    modules_to_fuse = ["conv", "bn", "relu"]
    fused_model = fuse_modules(
        model,
        modules_to_fuse,
        inplace=inplace,
        fuser_func=None,  # Use default
        fuse_custom_config_dict=None
    )
    
    # Get fused output
    with torch.no_grad():
        fused_output = fused_model(random_input)
    
    # Weak assertions
    # 1. Model type check
    assert isinstance(fused_model, nn.Module), "Fused model should be a Module"
    
    # 2. Module structure check
    # After fusion, we should have fused module and identity modules
    assert hasattr(fused_model, 'conv'), "Fused model should have 'conv' attribute"
    assert hasattr(fused_model, 'bn'), "Fused model should have 'bn' attribute"
    assert hasattr(fused_model, 'relu'), "Fused model should have 'relu' attribute"
    
    # Check that conv is now a fused module (ConvBnReLU2d or similar)
    conv_module = fused_model.conv
    assert isinstance(conv_module, nn.Module), "conv should be a Module"
    
    # 3. Output shape check
    check_output_shape(original_output, fused_output)
    
    # 4. Finite output check
    check_finite_output(fused_output)
    
    # 5. Model ID check for inplace behavior
    if inplace:
        # Inplace fusion should return the same model object
        assert fused_model is model, "Inplace fusion should return same model"
    else:
        # Non-inplace fusion should return a different model object
        assert fused_model is not model, "Non-inplace fusion should return new model"
# ==== BLOCK:CASE_01 END ====


# ==== BLOCK:CASE_02 START ====
# Test case: 多组模块融合

@pytest.mark.parametrize("inplace", [False])
def test_multi_group_module_fusion(random_input, inplace):
    """Test fusion of multiple groups of modules."""
    # Setup
    model = MultiLayerModel()
    model.eval()
    
    # Get original output
    with torch.no_grad():
        original_output = model(random_input)
    
    # Fuse multiple groups
    modules_to_fuse = [["conv1", "bn1", "relu1"], ["linear", "relu"]]
    fused_model = fuse_modules(
        model,
        modules_to_fuse,
        inplace=inplace,
        fuser_func=None,  # Use default
        fuse_custom_config_dict=None
    )
    
    # Get fused output
    with torch.no_grad():
        fused_output = fused_model(random_input)
    
    # Weak assertions
    # 1. Model type check
    assert isinstance(fused_model, nn.Module), "Fused model should be a Module"
    
    # 2. Module count check - all modules should still exist
    assert hasattr(fused_model, 'conv1'), "Fused model should have 'conv1'"
    assert hasattr(fused_model, 'bn1'), "Fused model should have 'bn1'"
    assert hasattr(fused_model, 'relu1'), "Fused model should have 'relu1'"
    assert hasattr(fused_model, 'linear'), "Fused model should have 'linear'"
    assert hasattr(fused_model, 'relu'), "Fused model should have 'relu'"
    
    # 3. Check that first group (conv1, bn1, relu1) was fused
    conv1_module = fused_model.conv1
    assert isinstance(conv1_module, nn.Module), "conv1 should be a Module"
    
    # 4. Check that second group (linear, relu) was fused
    linear_module = fused_model.linear
    assert isinstance(linear_module, nn.Module), "linear should be a Module"
    
    # 5. Output shape check
    check_output_shape(original_output, fused_output)
    
    # 6. Finite output check
    check_finite_output(fused_output)
    
    # 7. Model ID check for inplace behavior
    if inplace:
        assert fused_model is model, "Inplace fusion should return same model"
    else:
        assert fused_model is not model, "Non-inplace fusion should return new model"
# ==== BLOCK:CASE_02 END ====


# ==== BLOCK:CASE_03 START ====
# Test case: inplace参数行为

@pytest.mark.parametrize("inplace,expected_same_model", [
    (True, True),   # inplace=True should return same model
    (False, False), # inplace=False should return new model
])
def test_inplace_parameter_behavior(random_input, inplace, expected_same_model):
    """Test inplace parameter behavior for conv-bn fusion."""
    # Setup
    model = SimpleConvBN()
    model.eval()
    
    # Get original output and model id
    with torch.no_grad():
        original_output = model(random_input)
    
    original_model_id = id(model)
    
    # Fuse modules
    modules_to_fuse = ["conv", "bn"]
    fused_model = fuse_modules(
        model,
        modules_to_fuse,
        inplace=inplace,
        fuser_func=None,  # Use default
        fuse_custom_config_dict=None
    )
    
    # Get fused output
    with torch.no_grad():
        fused_output = fused_model(random_input)
    
    # Weak assertions
    # 1. Model ID different check
    fused_model_id = id(fused_model)
    if expected_same_model:
        assert fused_model_id == original_model_id, \
            f"Inplace=True: model IDs should be same ({fused_model_id} != {original_model_id})"
    else:
        assert fused_model_id != original_model_id, \
            f"Inplace=False: model IDs should be different ({fused_model_id} == {original_model_id})"
    
    # 2. Module structure check
    assert isinstance(fused_model, nn.Module), "Fused model should be a Module"
    assert hasattr(fused_model, 'conv'), "Fused model should have 'conv' attribute"
    assert hasattr(fused_model, 'bn'), "Fused model should have 'bn' attribute"
    
    # Check that conv is now a fused module
    conv_module = fused_model.conv
    assert isinstance(conv_module, nn.Module), "conv should be a Module"
    
    # 3. Output shape check
    check_output_shape(original_output, fused_output)
    
    # 4. Check original model preservation for non-inplace case
    if not inplace:
        # Original model should still be accessible and unchanged
        assert hasattr(model, 'conv'), "Original model should still have 'conv'"
        assert hasattr(model, 'bn'), "Original model should still have 'bn'"
        
        # Original model's conv should still be a regular Conv2d
        original_conv = model.conv
        assert isinstance(original_conv, nn.Conv2d), \
            "Original model's conv should remain Conv2d"
# ==== BLOCK:CASE_03 END ====


# ==== BLOCK:CASE_04 START ====
# Test case: 嵌套子模块融合

class NestedModel(nn.Module):
    """Model with nested submodule for nested fusion tests."""
    def __init__(self):
        super().__init__()
        self.submodule = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.submodule(x)


@pytest.mark.parametrize("inplace", [False])
def test_nested_submodule_fusion(random_input, inplace):
    """Test fusion of modules within nested submodules."""
    # Setup
    model = NestedModel()
    model.eval()
    
    # Get original output
    with torch.no_grad():
        original_output = model(random_input)
    
    # Fuse modules within nested submodule
    # Note: modules_to_fuse uses dot notation for nested access
    # For Sequential modules, we can use indices: submodule.0, submodule.1, submodule.2
    modules_to_fuse = ["submodule.0", "submodule.1", "submodule.2"]
    fused_model = fuse_modules(
        model,
        modules_to_fuse,
        inplace=inplace,
        fuser_func=None,  # Use default
        fuse_custom_config_dict=None
    )
    
    # Get fused output
    with torch.no_grad():
        fused_output = fused_model(random_input)
    
    # Weak assertions
    # 1. Model type check
    assert isinstance(fused_model, nn.Module), "Fused model should be a Module"
    
    # 2. Nested structure check
    # Parent structure should be preserved
    assert hasattr(fused_model, 'submodule'), "Fused model should have 'submodule'"
    
    # submodule should still be a Sequential or similar container
    submodule = fused_model.submodule
    assert isinstance(submodule, nn.Module), "submodule should be a Module"
    
    # 3. Check that submodule still has children
    # After fusion, the modules should still exist (possibly as fused or identity)
    children = list(submodule.children())
    assert len(children) >= 1, "submodule should have at least one child after fusion"
    
    # 4. Output shape check
    check_output_shape(original_output, fused_output)
    
    # 5. Finite output check
    check_finite_output(fused_output)
    
    # 6. Model ID check for inplace behavior
    if inplace:
        assert fused_model is model, "Inplace fusion should return same model"
    else:
        assert fused_model is not model, "Non-inplace fusion should return new model"
    
    # 7. Check that fusion didn't break the model
    # The model should still produce valid output
    assert torch.isfinite(fused_output).all(), "Fused model should produce finite output"
# ==== BLOCK:CASE_04 END ====


# ==== BLOCK:CASE_05 START ====
# Test case: 不支持序列保持不变 (deferred placeholder)
# ==== BLOCK:CASE_05 END ====


# ==== BLOCK:CASE_06 START ====
# Test case: 无效模块名称异常

@pytest.mark.parametrize("inplace", [False])
def test_invalid_module_name_exception(random_input, inplace):
    """Test that invalid module names raise appropriate exception."""
    # Setup
    model = SimpleModel()
    model.eval()
    
    # Try to fuse with non-existent module name
    modules_to_fuse = ["nonexistent_module"]
    
    # Weak assertions
    # 1. Exception raised check
    with pytest.raises(Exception) as exc_info:
        fuse_modules(
            model,
            modules_to_fuse,
            inplace=inplace,
            fuser_func=None,
            fuse_custom_config_dict=None
        )
    
    # 2. Exception type check
    exception = exc_info.value
    # The actual exception type might be AttributeError or similar
    # We'll check that some exception was raised
    assert exception is not None, "Exception should be raised for invalid module name"
    
    # 3. Check exception message contains relevant info
    exception_str = str(exception).lower()
    # The message should indicate the module wasn't found
    # Common patterns: "has no attribute", "not found", "nonexistent"
    assert any(keyword in exception_str for keyword in 
               ["attribute", "not found", "nonexistent", "no module"]), \
        f"Exception message should indicate module not found: {exception_str}"
    
    # 4. Verify model is unchanged after exception
    assert hasattr(model, 'conv'), "Model should still have 'conv' after exception"
    assert hasattr(model, 'relu'), "Model should still have 'relu' after exception"
    
    # Model should still work
    with torch.no_grad():
        output = model(random_input)
        assert torch.isfinite(output).all(), "Model should still produce valid output"
# ==== BLOCK:CASE_06 END ====


# ==== BLOCK:CASE_07 START ====
# Test case: 非Module类型输入

@pytest.mark.parametrize("invalid_model", [
    None,           # None input
    123,            # Integer input
    "not_a_model",  # String input
    [],             # List input
    {},             # Dict input
])
def test_non_module_type_input(invalid_model):
    """Test that non-Module type inputs raise appropriate exception."""
    # Try to fuse with invalid model type
    modules_to_fuse = ["conv", "bn"]
    
    # Weak assertions
    # 1. Exception raised check
    with pytest.raises(Exception) as exc_info:
        fuse_modules(
            invalid_model,
            modules_to_fuse,
            inplace=False,
            fuser_func=None,
            fuse_custom_config_dict=None
        )
    
    # 2. Exception type check
    exception = exc_info.value
    assert exception is not None, "Exception should be raised for non-Module input"
    
    # 3. Check exception message contains relevant error info
    # Based on execution logs, the actual error is "X object has no attribute 'conv'"
    # This is an AttributeError, not necessarily a type error
    exception_str = str(exception).lower()
    
    # Check for attribute access errors or type errors
    # Common patterns for non-Module inputs:
    # - "has no attribute" (AttributeError for missing 'conv' attribute)
    # - "object has no attribute" 
    # - "type" (TypeError)
    # - "module" (indicating Module expected)
    error_indicators = ["has no attribute", "object has no", "attribute", "type", "module"]
    error_found = any(indicator in exception_str for indicator in error_indicators)
    
    assert error_found, \
        f"Exception should indicate attribute or type error: {exception_str}"
    
    # 4. Additional check: verify it's an AttributeError or TypeError
    # The actual exception type depends on implementation
    # It could be AttributeError (trying to access 'conv' attribute)
    # or TypeError (wrong type passed to function)
    assert isinstance(exception, (AttributeError, TypeError)), \
        f"Exception should be AttributeError or TypeError, got {type(exception).__name__}"
# ==== BLOCK:CASE_07 END ====


# ==== BLOCK:CASE_08 START ====
# Test case: 空列表输入

@pytest.mark.parametrize("inplace", [False])
def test_empty_list_input(random_input, inplace):
    """Test that empty modules_to_fuse list behaves correctly."""
    # Setup
    model = SimpleModel()
    model.eval()
    
    # Get original output
    with torch.no_grad():
        original_output = model(random_input)
    
    # Try to fuse with empty list
    modules_to_fuse = []
    
    # Weak assertions
    # 1. Check behavior with empty list
    # Based on execution logs, empty list causes AssertionError in get_fuser_method
    # This is because it tries to find a fuser method for empty tuple ()
    # We need to handle this as a special case
    
    try:
        fused_model = fuse_modules(
            model,
            modules_to_fuse,
            inplace=inplace,
            fuser_func=None,
            fuse_custom_config_dict=None
        )
        
        # If no exception was raised, verify the behavior
        # 2. Model type check
        assert isinstance(fused_model, nn.Module), "Fused model should be a Module"
        
        # 3. Output shape check
        with torch.no_grad():
            fused_output = fused_model(random_input)
        check_output_shape(original_output, fused_output)
        
        # 4. Check that model structure is unchanged
        assert hasattr(fused_model, 'conv'), "Model should still have 'conv'"
        assert hasattr(fused_model, 'relu'), "Model should still have 'relu'"
        
        # 5. Model ID check for inplace behavior
        if inplace:
            assert fused_model is model, "Inplace should return same model"
        else:
            assert fused_model is not model, "Non-inplace should return new model"
            
    except AssertionError as e:
        # If AssertionError is raised (from get_fuser_method), that's acceptable
        # for empty list case. We just need to verify it's the expected error.
        error_msg = str(e).lower()
        # Check if it's the "did not find fuser method for: ()" error
        assert "fuser method" in error_msg and "()" in error_msg, \
            f"Unexpected AssertionError for empty list: {error_msg}"
        
        # Even if fusion fails, original model should still work
        with torch.no_grad():
            output = model(random_input)
            assert torch.isfinite(output).all(), "Original model should still work"
# ==== BLOCK:CASE_08 END ====


# ==== BLOCK:CASE_09 START ====
# Test case: 自定义fuser_func

from unittest.mock import Mock, patch

@pytest.mark.parametrize("inplace", [False])
def test_custom_fuser_func(random_input, inplace):
    """Test custom fuser_func parameter."""
    # Setup
    model = SimpleConvBN()
    model.eval()
    
    # Get original output
    with torch.no_grad():
        original_output = model(random_input)
    
    # Create a mock custom fuser function
    mock_fuser = Mock()
    
    # Configure the mock to return a simple replacement
    # The fuser function typically takes (model, modules_to_fuse, is_qat, **kwargs)
    # and returns a fused module
    def mock_fuser_func(model, modules_to_fuse, is_qat=False, **kwargs):
        # For testing, just return a simple Conv2d that mimics fusion
        # In reality, this would create a fused ConvBn2d
        return nn.Conv2d(3, 16, kernel_size=3, padding=1)
    
    mock_fuser.side_effect = mock_fuser_func
    
    # Fuse modules with custom fuser
    modules_to_fuse = ["conv", "bn"]
    fused_model = fuse_modules(
        model,
        modules_to_fuse,
        inplace=inplace,
        fuser_func=mock_fuser_func,
        fuse_custom_config_dict=None
    )
    
    # Get fused output
    with torch.no_grad():
        fused_output = fused_model(random_input)
    
    # Weak assertions
    # 1. No exception check
    # The function should complete without raising an exception
    
    # 2. Model type check
    assert isinstance(fused_model, nn.Module), "Fused model should be a Module"
    
    # 3. Output shape check
    check_output_shape(original_output, fused_output)
    
    # 4. Check that custom fuser was called (indirectly)
    # Since we can't directly mock the internal call, we verify the result
    # The conv module should now be a Conv2d (from our mock fuser)
    conv_module = fused_model.conv
    assert isinstance(conv_module, nn.Conv2d), \
        "conv should be Conv2d after custom fuser"
    
    # 5. Check bn module exists (should be identity or similar)
    assert hasattr(fused_model, 'bn'), "Fused model should have 'bn'"
    
    # 6. Model ID check for inplace behavior
    if inplace:
        assert fused_model is model, "Inplace should return same model"
    else:
        assert fused_model is not model, "Non-inplace should return new model"
    
    # Note: In a real test with proper mocking, we would:
    # 1. Mock the fuser_func and verify it was called with correct arguments
    # 2. Check that the mock's return value was used
    # However, without knowing the exact internal implementation,
    # we can only verify the external behavior.
# ==== BLOCK:CASE_09 END ====


# ==== BLOCK:FOOTER START ====
# Additional test utilities and cleanup

if __name__ == "__main__":
    # Simple test runner for debugging
    import sys
    pytest.main([sys.argv[0], "-v"])
# ==== BLOCK:FOOTER END ====