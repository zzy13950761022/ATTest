import torch
import pytest
import numpy as np
from torch.nn.utils import convert_parameters

# ==== BLOCK:HEADER START ====
import torch
import pytest
import numpy as np
from torch.nn.utils import convert_parameters

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Test file for torch.nn.utils.convert_parameters
# 
# This file contains tests for:
# - parameters_to_vector: converts parameters to a single vector
# - vector_to_parameters: converts a vector back to parameters
# 
# Test groups:
# - G1: parameters_to_vector function family
# - G2: vector_to_parameters function family
# 
# Current active group: G2 (vector_to_parameters)
# 
# Test plan based on:
# - SMOKE_SET: CASE_03, CASE_04 (G2)
# - DEFERRED_SET: CASE_07, CASE_08 (G2)
# 
# Epoch: 2/5 - Fixing HEADER block and G2 test cases
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
# Test case: CPU向量正常分割
# TC-03: CPU向量正常分割
# Priority: High
# Group: G2
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# Test case: 向量长度不匹配异常
# TC-04: 向量长度不匹配异常
# Priority: High
# Group: G2
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_07 START ====
# Test case: 非张量参数异常
# TC-07: 非张量参数异常
# Priority: Medium
# Group: G2
# Status: Deferred (placeholder)
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# Test case: 极端形状参数
# TC-08: 极端形状参数
# Priority: Medium
# Group: G2
# Status: Deferred (placeholder)
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Helper functions and fixtures

@pytest.fixture
def sample_parameters_cpu():
    """Fixture providing sample parameters on CPU for testing."""
    return [
        torch.randn(3, 2, dtype=torch.float32),
        torch.randn(5, dtype=torch.float32),
    ]

@pytest.fixture
def sample_parameters_gpu():
    """Fixture providing sample parameters on GPU for testing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    return [
        torch.randn(3, 2, dtype=torch.float32).cuda(),
        torch.randn(5, dtype=torch.float32).cuda(),
    ]

def create_parameters(shapes, dtype=torch.float32, device='cpu'):
    """Helper to create parameters with given shapes."""
    parameters = []
    for i, shape in enumerate(shapes):
        # Create tensor with unique values
        tensor = torch.arange(i * 10, i * 10 + np.prod(shape), dtype=dtype)
        tensor = tensor.reshape(shape)
        if device == 'cuda' and torch.cuda.is_available():
            tensor = tensor.cuda()
        parameters.append(tensor)
    return parameters

def verify_vector_parameters_roundtrip(parameters, rtol=1e-7, atol=1e-7):
    """Helper to verify round-trip conversion preserves values."""
    # Convert to vector
    vec = convert_parameters.parameters_to_vector(parameters)
    
    # Create new parameters with same shapes
    new_parameters = [torch.empty_like(p) for p in parameters]
    
    # Convert vector back to parameters
    convert_parameters.vector_to_parameters(vec, new_parameters)
    
    # Verify values are preserved
    for orig, new in zip(parameters, new_parameters):
        assert torch.allclose(orig, new, rtol=rtol, atol=atol), \
            "Round-trip conversion failed to preserve values"
    
    return vec, new_parameters

# Test class for better organization (optional)
class TestConvertParameters:
    """Test class for torch.nn.utils.convert_parameters module."""
    
    def test_import(self):
        """Test that the module can be imported."""
        from torch.nn.utils import convert_parameters
        assert hasattr(convert_parameters, 'parameters_to_vector')
        assert hasattr(convert_parameters, 'vector_to_parameters')
    
    def test_module_docstring(self):
        """Test that functions have docstrings."""
        from torch.nn.utils import convert_parameters
        assert convert_parameters.parameters_to_vector.__doc__ is not None
        assert convert_parameters.vector_to_parameters.__doc__ is not None
# ==== BLOCK:FOOTER END ====