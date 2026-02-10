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
# Current active group: G1 (parameters_to_vector)
# 
# Test plan based on:
# - SMOKE_SET: CASE_01, CASE_02 (G1)
# - DEFERRED_SET: CASE_05, CASE_06 (G1)
# 
# Epoch: 4/5 - Creating G1 test file
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
def test_parameters_to_vector_normal(shapes, device, dtype, requires_grad):
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

# ==== BLOCK:CASE_05 START ====
def test_parameters_to_vector_empty_iterable():
    """Test case: 空迭代器处理
    TC-05: 空迭代器处理
    Priority: Medium
    Group: G1
    Assertion level: weak (with strong assertions for final epoch)
    """
    # Test 1: Empty list
    print("Test 1: Empty list")
    empty_list = []
    
    # According to PyTorch implementation, empty iterable should work
    # It returns an empty tensor
    vec = convert_parameters.parameters_to_vector(empty_list)
    
    # Weak assertions
    assert isinstance(vec, torch.Tensor), "Should return a tensor"
    assert vec.dim() == 1, "Should be 1D tensor"
    assert vec.shape[0] == 0, "Should be empty tensor"
    
    # Test 2: Empty tuple
    print("Test 2: Empty tuple")
    empty_tuple = ()
    
    vec2 = convert_parameters.parameters_to_vector(empty_tuple)
    assert isinstance(vec2, torch.Tensor), "Should return a tensor"
    assert vec2.dim() == 1, "Should be 1D tensor"
    assert vec2.shape[0] == 0, "Should be empty tensor"
    
    # Test 3: Empty iterator
    print("Test 3: Empty iterator")
    def empty_iterator():
        return
        yield
    
    vec3 = convert_parameters.parameters_to_vector(empty_iterator())
    assert isinstance(vec3, torch.Tensor), "Should return a tensor"
    assert vec3.dim() == 1, "Should be 1D tensor"
    assert vec3.shape[0] == 0, "Should be empty tensor"
    
    # Test 4: Verify dtype and device of empty result
    print("Test 4: Verify dtype and device")
    # The empty tensor should have default dtype and device
    assert vec.dtype == torch.float32, f"Expected float32, got {vec.dtype}"
    assert vec.device.type == 'cpu', f"Expected CPU, got {vec.device}"
    
    # Test 5: Test with vector_to_parameters and empty parameters
    print("Test 5: vector_to_parameters with empty parameters")
    empty_vec = torch.tensor([], dtype=torch.float32)
    empty_params = []
    
    # This should work without error
    convert_parameters.vector_to_parameters(empty_vec, empty_params)
    
    # Strong assertions (enabled in final epoch)
    # 1. Verify the empty tensor properties
    assert vec.is_contiguous(), "Empty tensor should be contiguous"
    assert vec.storage().size() == 0, "Storage should be empty"
    
    # 2. Test edge case: empty parameters with non-empty vector
    print("Test 6: Empty parameters with non-empty vector")
    non_empty_vec = torch.randn(5, dtype=torch.float32)
    
    # This should work - vector elements will be ignored since there are no parameters
    convert_parameters.vector_to_parameters(non_empty_vec, [])
    
    # 3. Test with generator that yields nothing
    print("Test 7: Generator that yields nothing")
    def no_yield():
        if False:
            yield torch.tensor([1.0])
    
    vec4 = convert_parameters.parameters_to_vector(no_yield())
    assert vec4.shape[0] == 0, "Should be empty tensor"
    
    # 4. Test memory layout
    print("Test 8: Memory layout")
    # Create a non-empty tensor first to ensure memory is allocated
    non_empty = [torch.tensor([1.0, 2.0, 3.0])]
    vec_non_empty = convert_parameters.parameters_to_vector(non_empty)
    
    # Then test empty
    vec_empty = convert_parameters.parameters_to_vector([])
    
    # They should both be valid tensors
    assert vec_non_empty.numel() == 3
    assert vec_empty.numel() == 0
    
    # 5. Test with mixed empty and non-empty (should not happen in practice)
    print("Test 9: Round-trip with empty")
    # Convert empty to vector
    vec_empty2 = convert_parameters.parameters_to_vector([])
    # Convert back (nothing to convert back to)
    convert_parameters.vector_to_parameters(vec_empty2, [])
    
    # This should not crash
    assert True, "Round-trip with empty should work"
    
    # 6. Test error handling for invalid empty cases
    print("Test 10: Invalid cases with empty")
    # None is not an iterable
    with pytest.raises(TypeError):
        convert_parameters.parameters_to_vector(None)
    
    # String is iterable but yields characters, not tensors
    with pytest.raises(TypeError):
        convert_parameters.parameters_to_vector("abc")
    
    # Empty string still yields characters (0 of them)
    with pytest.raises(TypeError):
        convert_parameters.parameters_to_vector("")
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
@pytest.mark.parametrize("shapes,device,dtype,requires_grad", [
    ([[0], [2, 0, 3]], 'cpu', torch.float32, False),  # Original test case
])
def test_parameters_to_vector_zero_elements(shapes, device, dtype, requires_grad):
    """Test case: 零元素参数处理
    TC-06: 零元素参数处理
    Priority: Medium
    Group: G1
    Assertion level: weak (with strong assertions for final epoch)
    """
    # Skip if CUDA is not available
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create parameters with zero elements
    parameters = []
    total_elements = 0
    
    for i, shape in enumerate(shapes):
        num_elements = np.prod(shape)
        
        # Create empty tensor
        if num_elements == 0:
            # Zero-dimensional or shape with zero
            if device == 'cuda:0' and torch.cuda.is_available():
                tensor = torch.empty(shape, dtype=dtype, device='cuda')
            else:
                tensor = torch.empty(shape, dtype=dtype)
        else:
            # Should not happen with our test cases, but handle anyway
            if dtype in [torch.float32, torch.float64]:
                tensor = torch.arange(i * 10, i * 10 + num_elements, dtype=dtype)
                tensor = tensor.float() / 10.0
            else:
                tensor = torch.arange(i * 10, i * 10 + num_elements, dtype=dtype)
            tensor = tensor.reshape(shape)
            
            if device == 'cuda:0' and torch.cuda.is_available():
                tensor = tensor.cuda()
        
        # Set requires_grad if needed
        if requires_grad:
            tensor.requires_grad_(True)
        
        parameters.append(tensor)
        total_elements += tensor.numel()
    
    print(f"Testing zero-element shapes: {shapes}, total elements: {total_elements}")
    
    # Weak assertions
    # 1. Convert parameters to vector
    vec = convert_parameters.parameters_to_vector(parameters)
    
    # Should be 1D tensor with zero elements
    assert vec.dim() == 1, f"Expected 1D tensor, got shape {vec.shape}"
    assert vec.shape[0] == total_elements, \
        f"Expected length {total_elements}, got {vec.shape[0]}"
    
    # 2. Device match
    if total_elements > 0:
        expected_device = parameters[0].device
        assert vec.device == expected_device, \
            f"Expected device {expected_device}, got {vec.device}"
    
    # 3. Dtype match
    if total_elements > 0:
        expected_dtype = parameters[0].dtype
        assert vec.dtype == expected_dtype, \
            f"Expected dtype {expected_dtype}, got {vec.dtype}"
    
    # 4. Test round-trip conversion
    if total_elements > 0:
        # Create new empty parameters
        new_parameters = []
        for shape in shapes:
            if device == 'cuda:0' and torch.cuda.is_available():
                new_param = torch.empty(shape, dtype=dtype, device='cuda')
            else:
                new_param = torch.empty(shape, dtype=dtype)
            
            if requires_grad:
                new_param.requires_grad_(True)
            
            new_parameters.append(new_param)
        
        # Convert vector back to parameters
        convert_parameters.vector_to_parameters(vec, new_parameters)
        
        # Verify shapes are preserved
        for orig, new in zip(parameters, new_parameters):
            assert orig.shape == new.shape, \
                f"Shape mismatch: expected {orig.shape}, got {new.shape}"
    
    # Strong assertions (enabled in final epoch)
    # 1. Edge case consistency: all zero-element parameters
    all_zero_shapes = [[0], [0, 0], [1, 0, 2], [0, 3, 0]]
    zero_params = []
    
    for shape in all_zero_shapes:
        tensor = torch.empty(shape, dtype=dtype)
        zero_params.append(tensor)
    
    zero_vec = convert_parameters.parameters_to_vector(zero_params)
    assert zero_vec.shape[0] == 0, "All zero-element parameters should produce empty vector"
    
    # 2. Mixed zero and non-zero elements
    mixed_shapes = [[0], [2, 3], [1, 0, 2], [4]]
    mixed_params = []
    
    for i, shape in enumerate(mixed_shapes):
        num_elements = np.prod(shape)
        if num_elements > 0:
            tensor = torch.arange(i * 10, i * 10 + num_elements, dtype=dtype)
            if dtype in [torch.float32, torch.float64]:
                tensor = tensor.float() / 10.0
            tensor = tensor.reshape(shape)
        else:
            tensor = torch.empty(shape, dtype=dtype)
        mixed_params.append(tensor)
    
    mixed_vec = convert_parameters.parameters_to_vector(mixed_params)
    
    # Calculate expected total
    expected_total = sum(np.prod(s) for s in mixed_shapes)
    assert mixed_vec.shape[0] == expected_total, \
        f"Expected {expected_total} elements, got {mixed_vec.shape[0]}"
    
    # 3. Verify concatenation order with zero elements
    pointer = 0
    for i, param in enumerate(mixed_params):
        num_elements = param.numel()
        if num_elements > 0:
            param_flat = param.view(-1)
            vec_slice = mixed_vec[pointer:pointer + num_elements]
            
            if dtype in [torch.float32, torch.float64]:
                assert torch.allclose(param_flat, vec_slice, rtol=1e-7, atol=1e-7), \
                    f"Values not preserved for parameter {i}"
            else:
                assert torch.equal(param_flat, vec_slice), \
                    f"Values not preserved for parameter {i}"
        
        pointer += num_elements
    
    # 4. Test vector_to_parameters with zero-element parameters
    zero_targets = [torch.empty_like(p) for p in mixed_params]
    convert_parameters.vector_to_parameters(mixed_vec, zero_targets)
    
    # Verify non-zero parameters were filled
    for orig, new in zip(mixed_params, zero_targets):
        if orig.numel() > 0:
            if dtype in [torch.float32, torch.float64]:
                assert torch.allclose(orig, new, rtol=1e-7, atol=1e-7), \
                    "Round-trip failed for mixed zero/non-zero parameters"
            else:
                assert torch.equal(orig, new), \
                    "Round-trip failed for mixed zero/non-zero parameters"
    
    # 5. Memory and performance checks
    # Test with many zero-element parameters
    many_zero_params = [torch.empty(0, dtype=dtype) for _ in range(100)]
    many_zero_vec = convert_parameters.parameters_to_vector(many_zero_params)
    assert many_zero_vec.shape[0] == 0, "Many zero-element params should produce empty vector"
    
    # 6. Test with requires_grad on zero-element tensors
    zero_with_grad = torch.empty(0, dtype=dtype, requires_grad=True)
    vec_grad = convert_parameters.parameters_to_vector([zero_with_grad])
    assert vec_grad.shape[0] == 0, "Zero-element tensor with grad should produce empty vector"
    
    # 7. Test device consistency with zero-element tensors
    if torch.cuda.is_available():
        zero_cpu = torch.empty(0, dtype=dtype)
        zero_gpu = torch.empty(0, dtype=dtype, device='cuda')
        
        # Mixing devices should still fail even with zero-element tensors
        with pytest.raises(TypeError) as exc_info:
            convert_parameters.parameters_to_vector([zero_cpu, zero_gpu])
        
        error_msg = str(exc_info.value).lower()
        assert "device" in error_msg or "different" in error_msg, \
            f"Expected device error even with zero-element tensors, got: {error_msg}"
    
    # 8. Test extreme case: shape with product zero but non-zero dimensions
    extreme_shapes = [[1000, 0], [0, 1000], [10, 0, 10]]
    extreme_params = [torch.empty(s, dtype=dtype) for s in extreme_shapes]
    extreme_vec = convert_parameters.parameters_to_vector(extreme_params)
    assert extreme_vec.shape[0] == 0, "All zero-element tensors should produce empty vector"
# ==== BLOCK:CASE_06 END ====

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
        if dtype in [torch.float32, torch.float64]:
            tensor = torch.arange(i * 10, i * 10 + np.prod(shape), dtype=dtype)
            tensor = tensor.float() / 10.0
        else:
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
        if orig.dtype in [torch.float32, torch.float64]:
            assert torch.allclose(orig, new, rtol=rtol, atol=atol), \
                "Round-trip conversion failed to preserve values"
        else:
            assert torch.equal(orig, new), \
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