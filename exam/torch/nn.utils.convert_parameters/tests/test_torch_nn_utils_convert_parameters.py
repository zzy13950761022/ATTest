import torch
import pytest
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
# Current active group: G1 (parameters_to_vector)
# 
# Test plan based on:
# - SMOKE_SET: CASE_01, CASE_02 (G1), CASE_03, CASE_04 (G2)
# - DEFERRED_SET: CASE_05, CASE_06 (G1), CASE_07, CASE_08 (G2)
# 
# Epoch: 1/5 - First round with weak assertions only
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize("shapes,device,dtype,requires_grad", [
    # Original test case
    ([[2, 3], [4], [1, 5, 2]], 'cpu', torch.float32, False),
    # Medium priority extension: GPU device with float64 and requires_grad
    ([[4, 4], [2, 3, 2]], 'cuda:0', torch.float64, True),
    # Low priority extension: integer type with minimal shapes
    ([[1], [1, 1, 1]], 'cpu', torch.int64, False),
])
def test_parameters_to_vector_cpu_normal(shapes, device, dtype, requires_grad):
    """Test case: CPU参数正常展平
    TC-01: CPU参数正常展平
    Priority: High with extensions
    Group: G1
    Assertion level: weak
    """
    # Skip if CUDA is not available
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create parameters with different shapes
    parameters = []
    total_elements = 0
    
    for i, shape in enumerate(shapes):
        # Create tensor with unique values for easy verification
        if dtype in [torch.float32, torch.float64]:
            # For float types, use float values
            tensor = torch.arange(i * 10, i * 10 + np.prod(shape), dtype=dtype)
            tensor = tensor.float() / 10.0  # Make them float values
        else:
            # For integer types, use integer values
            tensor = torch.arange(i * 10, i * 10 + np.prod(shape), dtype=dtype)
        
        tensor = tensor.reshape(shape)
        
        # Move to device if needed
        if device == 'cuda:0':
            tensor = tensor.cuda()
        
        # Set requires_grad if needed
        if requires_grad:
            tensor.requires_grad_(True)
        
        parameters.append(tensor)
        total_elements += tensor.numel()
    
    # Convert parameters to vector
    vec = convert_parameters.parameters_to_vector(parameters)
    
    # Weak assertions
    # 1. Shape match: should be 1D tensor with length equal to total elements
    assert vec.dim() == 1, f"Expected 1D tensor, got shape {vec.shape}"
    assert vec.shape[0] == total_elements, \
        f"Expected length {total_elements}, got {vec.shape[0]}"
    
    # 2. Device match: should be on same device as parameters
    expected_device = parameters[0].device
    assert vec.device == expected_device, \
        f"Expected device {expected_device}, got {vec.device}"
    
    # 3. Dtype match: should be same dtype as parameters
    expected_dtype = parameters[0].dtype
    assert vec.dtype == expected_dtype, \
        f"Expected dtype {expected_dtype}, got {vec.dtype}"
    
    # 4. Values preserved: check concatenation order
    pointer = 0
    for i, param in enumerate(parameters):
        num_elements = param.numel()
        # Flatten the parameter for comparison
        param_flat = param.view(-1)
        vec_slice = vec[pointer:pointer + num_elements]
        
        # Check values are preserved
        # Use appropriate tolerance for float types
        if dtype in [torch.float32, torch.float64]:
            assert torch.allclose(param_flat, vec_slice, rtol=1e-7, atol=1e-7), \
                f"Values not preserved for parameter {i}"
        else:
            # For integer types, exact match
            assert torch.equal(param_flat, vec_slice), \
                f"Values not preserved for parameter {i}"
        
        pointer += num_elements
    
    # Additional check: verify the entire vector
    expected_vec = torch.cat([p.view(-1) for p in parameters])
    if dtype in [torch.float32, torch.float64]:
        assert torch.allclose(vec, expected_vec, rtol=1e-7, atol=1e-7), \
            "Vector does not match manual concatenation"
    else:
        assert torch.equal(vec, expected_vec), \
            "Vector does not match manual concatenation"
    
    # Check requires_grad is preserved for the vector
    if requires_grad:
        assert vec.requires_grad, "Vector should require grad when parameters do"
    else:
        assert not vec.requires_grad, "Vector should not require grad when parameters don't"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("devices,shapes", [
    (["cpu", "cuda:0"], [[2, 2], [3]]),  # Original test case
])
def test_parameters_to_vector_device_mismatch(devices, shapes):
    """Test case: 设备不一致异常
    TC-02: 设备不一致异常
    Priority: High
    Group: G1
    Assertion level: weak
    """
    # Skip if CUDA is not available
    if "cuda:0" in devices and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create parameters on different devices
    parameters = []
    
    for i, (device_str, shape) in enumerate(zip(devices, shapes)):
        # Create tensor
        tensor = torch.randn(shape, dtype=torch.float32)
        
        # Move to specified device
        if device_str == "cuda:0":
            tensor = tensor.cuda()
        
        parameters.append(tensor)
    
    # Weak assertions: should raise TypeError
    with pytest.raises(TypeError) as exc_info:
        convert_parameters.parameters_to_vector(parameters)
    
    # Check exception message contains expected keywords
    error_msg = str(exc_info.value).lower()
    assert "device" in error_msg or "different" in error_msg, \
        f"Expected device-related error, got: {error_msg}"
    
    # Additional check for specific error message pattern
    # The actual error message is: "Found two parameters on different devices, this is currently not supported."
    assert "different devices" in error_msg or "not supported" in error_msg, \
        f"Error message should mention device mismatch, got: {error_msg}"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize("shapes", [
    ([[3, 2], [5]],),  # Original test case
])
def test_vector_to_parameters_cpu_normal(shapes):
    """Test case: CPU向量正常分割
    TC-03: CPU向量正常分割
    Priority: High
    Group: G2
    Assertion level: weak
    """
    # Unpack shapes
    shapes = shapes[0]
    
    # Create original parameters with unique values
    original_parameters = []
    total_elements = 0
    
    for i, shape in enumerate(shapes):
        # Create tensor with unique values for easy verification
        tensor = torch.arange(i * 10, i * 10 + np.prod(shape), dtype=torch.float32)
        tensor = tensor.reshape(shape)
        original_parameters.append(tensor)
        total_elements += tensor.numel()
    
    # Convert original parameters to vector
    vec = convert_parameters.parameters_to_vector(original_parameters)
    
    # Create empty target parameters with same shapes
    target_parameters = []
    for shape in shapes:
        target_parameters.append(torch.empty(shape, dtype=torch.float32))
    
    # Convert vector back to target parameters
    convert_parameters.vector_to_parameters(vec, target_parameters)
    
    # Weak assertions
    # 1. Shape restored: target parameters should have correct shapes
    for i, (orig, target) in enumerate(zip(original_parameters, target_parameters)):
        assert target.shape == orig.shape, \
            f"Parameter {i}: expected shape {orig.shape}, got {target.shape}"
    
    # 2. Device preserved: should be on CPU
    for i, target in enumerate(target_parameters):
        assert target.device.type == 'cpu', \
            f"Parameter {i}: expected CPU device, got {target.device}"
    
    # 3. Dtype preserved: should be float32
    for i, target in enumerate(target_parameters):
        assert target.dtype == torch.float32, \
            f"Parameter {i}: expected float32, got {target.dtype}"
    
    # 4. Values preserved: target parameters should match original
    for i, (orig, target) in enumerate(zip(original_parameters, target_parameters)):
        assert torch.allclose(orig, target, rtol=1e-7, atol=1e-7), \
            f"Parameter {i}: values not preserved"
    
    # Additional check: verify the vector was correctly partitioned
    # Reconstruct the vector from target parameters
    reconstructed_vec = torch.cat([p.view(-1) for p in target_parameters])
    assert torch.allclose(vec, reconstructed_vec, rtol=1e-7, atol=1e-7), \
        "Vector reconstruction failed"
    
    # Test in-place modification: modify vector and verify parameters are updated
    modified_vec = vec.clone()
    modified_vec[0] = 999.0  # Modify first element
    
    # Convert modified vector to parameters
    convert_parameters.vector_to_parameters(modified_vec, target_parameters)
    
    # Verify the modification is reflected in the first parameter
    assert torch.allclose(target_parameters[0].view(-1)[0], torch.tensor(999.0)), \
        "In-place modification not reflected in parameters"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
@pytest.mark.parametrize("shapes,vec_length_mismatch", [
    ([[2, 2], [3]], True),  # Original test case with length mismatch
])
def test_vector_to_parameters_length_mismatch(shapes, vec_length_mismatch):
    """Test case: 向量长度不匹配异常
    TC-04: 向量长度不匹配异常
    Priority: High
    Group: G2
    Assertion level: weak
    """
    # Skip if not testing length mismatch
    if not vec_length_mismatch:
        pytest.skip("Not testing length mismatch scenario")
    
    # Create parameters with given shapes
    parameters = []
    total_elements = 0
    
    for i, shape in enumerate(shapes):
        # Create empty tensors with the given shapes
        tensor = torch.empty(shape, dtype=torch.float32)
        parameters.append(tensor)
        total_elements += tensor.numel()
    
    # Create a vector that is shorter than needed
    # Total elements needed: 2*2 + 3 = 7 elements
    # Create vector with only 5 elements to cause mismatch
    vec = torch.randn(5, dtype=torch.float32)  # Only 5 elements, need 7
    
    # Weak assertions: should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        convert_parameters.vector_to_parameters(vec, parameters)
    
    # Check exception message
    error_msg = str(exc_info.value).lower()
    
    # The actual error from PyTorch is "shape [3] is invalid for input of size 2"
    # This happens because when trying to view the slice as the parameter shape,
    # the slice is too short for the shape.
    
    # Check for expected error patterns
    # The error could be about shape mismatch or invalid size
    assert "shape" in error_msg or "size" in error_msg or "invalid" in error_msg, \
        f"Expected shape/size/invalid error, got: {error_msg}"
    
    # Additional check: verify it's not a device mismatch error
    assert "device" not in error_msg, \
        f"Expected length mismatch error, got device error: {error_msg}"
    
    # Test another scenario: vector longer than needed
    # This should also raise an error when trying to assign to the last parameter
    vec_long = torch.randn(10, dtype=torch.float32)  # 10 elements, need 7
    
    with pytest.raises(RuntimeError) as exc_info2:
        convert_parameters.vector_to_parameters(vec_long, parameters)
    
    error_msg2 = str(exc_info2.value).lower()
    assert "shape" in error_msg2 or "size" in error_msg2 or "invalid" in error_msg2, \
        f"Expected shape/size/invalid error for long vector, got: {error_msg2}"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# Test case: 空迭代器处理
# TC-05: 空迭代器处理
# Priority: Medium
# Group: G1
# Status: Deferred (placeholder)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# Test case: 零元素参数处理
# TC-06: 零元素参数处理
# Priority: Medium
# Group: G1
# Status: Deferred (placeholder)
# ==== BLOCK:CASE_06 END ====

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
        torch.randn(2, 3, dtype=torch.float32),
        torch.randn(4, dtype=torch.float32),
        torch.randn(1, 5, 2, dtype=torch.float32),
    ]

@pytest.fixture
def sample_parameters_gpu():
    """Fixture providing sample parameters on GPU for testing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    return [
        torch.randn(2, 3, dtype=torch.float32).cuda(),
        torch.randn(4, dtype=torch.float32).cuda(),
        torch.randn(1, 5, 2, dtype=torch.float32).cuda(),
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