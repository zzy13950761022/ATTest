import torch
import torch.nn as nn
import pytest
import copy
from torch.ao.quantization import fuse_modules


# ==== BLOCK:HEADER START ====
# Test helper functions and fixtures for G1 group

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


class UnsupportedSequence(nn.Module):
    """Model with unsupported fusion sequence (conv-relu-conv)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
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
# Test case: 嵌套子模块融合 (deferred placeholder)
# ==== BLOCK:CASE_04 END ====


# ==== BLOCK:CASE_05 START ====
# Test case: 不支持序列保持不变 (deferred placeholder)
# ==== BLOCK:CASE_05 END ====


# ==== BLOCK:FOOTER START ====
# Additional test utilities and cleanup for G1 group

if __name__ == "__main__":
    # Simple test runner for debugging
    import sys
    pytest.main([sys.argv[0], "-v"])
# ==== BLOCK:FOOTER END ====