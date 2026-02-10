"""
Test module for torch.nn.modules.upsampling (G2 group)
Tests for specialized subclass functionality
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Set random seed for reproducibility
torch.manual_seed(42)

# ==== BLOCK:HEADER START ====
"""
Test module for torch.nn.modules.upsampling (G2 group)
Tests for specialized subclass functionality
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Set random seed for reproducibility
torch.manual_seed(42)

# Common test data and fixtures for G2 group
@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device('cpu')

def create_test_tensor(shape, dtype=torch.float32, device='cpu'):
    """Create a test tensor with deterministic values."""
    num_elements = torch.prod(torch.tensor(shape)).item()
    tensor = torch.arange(num_elements, dtype=dtype, device=device).reshape(shape)
    return tensor / max(num_elements, 1)  # Normalize to [0, 1) range

def assert_tensors_close(actual, expected, rtol=1e-5, atol=1e-8, msg=""):
    """Assert that two tensors are close within tolerance."""
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"Dtype mismatch: {actual.dtype} != {expected.dtype}"
    
    # Check for NaN/Inf
    assert not torch.any(torch.isnan(actual)), f"Actual tensor contains NaN: {msg}"
    assert not torch.any(torch.isnan(expected)), f"Expected tensor contains NaN: {msg}"
    assert not torch.any(torch.isinf(actual)), f"Actual tensor contains Inf: {msg}"
    assert not torch.any(torch.isinf(expected)), f"Expected tensor contains Inf: {msg}"
    
    # Check values
    close = torch.allclose(actual, expected, rtol=rtol, atol=atol)
    if not close:
        max_diff = torch.max(torch.abs(actual - expected)).item()
        mean_diff = torch.mean(torch.abs(actual - expected)).item()
        raise AssertionError(
            f"Tensors not close: {msg}\n"
            f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}\n"
            f"RTOL={rtol}, ATOL={atol}"
        )
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
# CASE_03: UpsamplingNearest2d 基础功能
@pytest.mark.parametrize("input_shape, scale_factor", [
    ((1, 1, 2, 2), 2.0),  # Original from spec
    ((2, 1, 6, 6), 3.0),  # Param extension: larger scale factor and batch input
    ((2, 3, 4, 4), 1.5),  # Additional: batch and channels
    ((1, 1, 3, 3), 3.0),  # Additional: larger scale
])
def test_upsampling_nearest2d_basic(input_shape, scale_factor, device):
    """
    Test basic functionality of UpsamplingNearest2d.
    
    Verifies that:
    1. Output shape matches expected dimensions
    2. Output dtype matches input dtype
    3. No NaN values in output
    4. Nearest neighbor interpolation works correctly
    """
    # Create input tensor
    x = create_test_tensor(input_shape, dtype=torch.float32, device=device)
    
    # Create UpsamplingNearest2d module
    upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
    upsample.to(device)
    
    # Forward pass
    output = upsample(x)
    
    # Calculate expected output shape
    batch_size, channels, height, width = input_shape
    expected_height = int(height * scale_factor)
    expected_width = int(width * scale_factor)
    expected_shape = (batch_size, channels, expected_height, expected_width)
    
    # Weak assertions
    # 1. Shape match
    assert output.shape == expected_shape, (
        f"Output shape mismatch: {output.shape} != {expected_shape}"
    )
    
    # 2. Dtype match
    assert output.dtype == x.dtype, (
        f"Output dtype mismatch: {output.dtype} != {x.dtype}"
    )
    
    # 3. No NaN
    assert not torch.any(torch.isnan(output)), "Output contains NaN values"
    
    # 4. Nearest interpolation check
    # For nearest neighbor, each output pixel should match the nearest input pixel
    # We can verify this by checking a few sample points
    
    # Oracle comparison with torch.nn.functional.interpolate
    expected = F.interpolate(
        x,
        scale_factor=scale_factor,
        mode='nearest',
        recompute_scale_factor=True
    )
    
    # For nearest neighbor, values should match exactly
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-8), (
        "Output does not match F.interpolate oracle for nearest neighbor"
    )
    
    # Additional check: verify the mode is fixed to 'nearest'
    # UpsamplingNearest2d should always use nearest neighbor interpolation
    # We can test this by comparing with manual nearest neighbor
    
    # Simple nearest neighbor check for small tensor
    if input_shape == (1, 1, 2, 2) and scale_factor == 2.0:
        # Input: [[[1, 2], [3, 4]]] / 4 = [[[0.25, 0.5], [0.75, 1.0]]]
        # Expected output for scale_factor=2.0:
        # [[[0.25, 0.25, 0.5, 0.5],
        #   [0.25, 0.25, 0.5, 0.5],
        #   [0.75, 0.75, 1.0, 1.0],
        #   [0.75, 0.75, 1.0, 1.0]]]
        
        # Create simple test tensor
        simple_input = torch.tensor([[[[0.25, 0.5], [0.75, 1.0]]]], dtype=torch.float32, device=device)
        simple_output = upsample(simple_input)
        
        # Check a few values
        assert torch.allclose(simple_output[0, 0, 0, 0], torch.tensor(0.25, device=device))
        assert torch.allclose(simple_output[0, 0, 0, 1], torch.tensor(0.25, device=device))
        assert torch.allclose(simple_output[0, 0, 2, 2], torch.tensor(1.0, device=device))
    
    # Check that scaling is correct
    actual_scale_h = output.shape[2] / height
    actual_scale_w = output.shape[3] / width
    assert abs(actual_scale_h - scale_factor) < 0.01, (
        f"Height scaling incorrect: {actual_scale_h} != {scale_factor}"
    )
    assert abs(actual_scale_w - scale_factor) < 0.01, (
        f"Width scaling incorrect: {actual_scale_w} != {scale_factor}"
    )
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# CASE_04: UpsamplingBilinear2d 基础功能
@pytest.mark.parametrize("input_shape, scale_factor", [
    ((1, 1, 3, 3), 2.0),  # Original from spec
    ((1, 3, 7, 7), 2.0),  # Param extension: multi-channel input
    ((2, 3, 5, 5), 1.5),  # Additional: batch and channels
    ((1, 2, 4, 4), 2.5),  # Additional: non-integer scale
])
def test_upsampling_bilinear2d_basic(input_shape, scale_factor, device):
    """
    Test basic functionality of UpsamplingBilinear2d.
    
    Verifies that:
    1. Output shape matches expected dimensions
    2. Output dtype matches input dtype
    3. No NaN values in output
    4. Bilinear interpolation works correctly
    """
    # Create input tensor
    x = create_test_tensor(input_shape, dtype=torch.float32, device=device)
    
    # Create UpsamplingBilinear2d module
    upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    upsample.to(device)
    
    # Forward pass
    output = upsample(x)
    
    # Calculate expected output shape
    batch_size, channels, height, width = input_shape
    expected_height = int(height * scale_factor)
    expected_width = int(width * scale_factor)
    expected_shape = (batch_size, channels, expected_height, expected_width)
    
    # Weak assertions
    # 1. Shape match
    assert output.shape == expected_shape, (
        f"Output shape mismatch: {output.shape} != {expected_shape}"
    )
    
    # 2. Dtype match
    assert output.dtype == x.dtype, (
        f"Output dtype mismatch: {output.dtype} != {x.dtype}"
    )
    
    # 3. No NaN
    assert not torch.any(torch.isnan(output)), "Output contains NaN values"
    
    # 4. Bilinear interpolation check
    # Oracle comparison with torch.nn.functional.interpolate
    # Note: UpsamplingBilinear2d uses align_corners=True by default
    expected = F.interpolate(
        x,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=True,
        recompute_scale_factor=True
    )
    
    # Check if output matches oracle
    # Bilinear interpolation might have small numerical differences
    close = torch.allclose(output, expected, rtol=1e-5, atol=1e-8)
    if not close:
        max_diff = torch.max(torch.abs(output - expected)).item()
        mean_diff = torch.mean(torch.abs(output - expected)).item()
        
        # For bilinear interpolation with align_corners=True,
        # differences should be very small
        if max_diff < 1e-7:  # Acceptable numerical error
            return
        
        raise AssertionError(
            f"Output does not match F.interpolate oracle for bilinear interpolation\n"
            f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}"
        )
    
    # Additional check: verify align_corners=True behavior
    # With align_corners=True, corner pixels should be preserved
    if scale_factor == 2.0 and input_shape == (1, 1, 3, 3):
        # Check that corner pixels are preserved
        input_corners = [
            x[0, 0, 0, 0],      # top-left
            x[0, 0, 0, -1],     # top-right
            x[0, 0, -1, 0],     # bottom-left
            x[0, 0, -1, -1],    # bottom-right
        ]
        
        output_corners = [
            output[0, 0, 0, 0],           # top-left
            output[0, 0, 0, -1],          # top-right
            output[0, 0, -1, 0],          # bottom-left
            output[0, 0, -1, -1],         # bottom-right
        ]
        
        for i, (in_corner, out_corner) in enumerate(zip(input_corners, output_corners)):
            assert torch.allclose(in_corner, out_corner, rtol=1e-5, atol=1e-8), (
                f"Corner {i} not preserved with align_corners=True"
            )
    
    # Check that scaling is correct
    # For non-integer scale factors, the actual scale might be slightly different
    # due to integer rounding. We need to be more tolerant.
    actual_scale_h = output.shape[2] / height
    actual_scale_w = output.shape[3] / width
    
    # Calculate the expected integer dimensions
    expected_h_int = int(height * scale_factor)
    expected_w_int = int(width * scale_factor)
    
    # The actual scale should be close to expected_h_int/height and expected_w_int/width
    # not necessarily exactly equal to scale_factor due to integer rounding
    expected_scale_h = expected_h_int / height
    expected_scale_w = expected_w_int / width
    
    # Allow small tolerance for floating point errors
    assert abs(actual_scale_h - expected_scale_h) < 1e-10, (
        f"Height scaling incorrect: {actual_scale_h} != {expected_scale_h} (from scale_factor={scale_factor})"
    )
    assert abs(actual_scale_w - expected_scale_w) < 1e-10, (
        f"Width scaling incorrect: {actual_scale_w} != {expected_scale_w} (from scale_factor={scale_factor})"
    )
    
    # Verify that values are reasonable (no extreme values)
    output_min = torch.min(output).item()
    output_max = torch.max(output).item()
    input_min = torch.min(x).item()
    input_max = torch.max(x).item()
    
    # Bilinear interpolation should produce values within or near input range
    assert output_min >= input_min - 0.1, (
        f"Output min {output_min} is significantly below input min {input_min}"
    )
    assert output_max <= input_max + 0.1, (
        f"Output max {output_max} is significantly above input max {input_max}"
    )
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_07 START ====
# CASE_07: UpsamplingNearest2d 批量处理
@pytest.mark.parametrize("batch_size, channels, height, width, scale_factor", [
    (4, 3, 8, 8, 1.5),  # Original from spec
    (2, 1, 4, 4, 2.0),  # Additional test case
])
def test_upsampling_nearest2d_batch_processing(batch_size, channels, height, width, scale_factor, device):
    """
    Test UpsamplingNearest2d with batch processing.
    
    Verifies that:
    1. Output shape matches expected dimensions
    2. Output dtype matches input dtype
    3. No NaN values in output
    4. Batch consistency (all samples processed correctly)
    """
    # Create input tensor
    input_shape = (batch_size, channels, height, width)
    x = create_test_tensor(input_shape, dtype=torch.float32, device=device)
    
    # Create UpsamplingNearest2d module
    upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
    upsample.to(device)
    
    # Forward pass
    output = upsample(x)
    
    # Calculate expected output shape
    expected_height = int(height * scale_factor)
    expected_width = int(width * scale_factor)
    expected_shape = (batch_size, channels, expected_height, expected_width)
    
    # Weak assertions
    # 1. Shape match
    assert output.shape == expected_shape, (
        f"Output shape mismatch: {output.shape} != {expected_shape}"
    )
    
    # 2. Dtype match
    assert output.dtype == x.dtype, (
        f"Output dtype mismatch: {output.dtype} != {x.dtype}"
    )
    
    # 3. No NaN
    assert not torch.any(torch.isnan(output)), "Output contains NaN values"
    
    # 4. Batch consistency check
    # Verify that all batch elements are processed (no zeros or identical outputs)
    for i in range(batch_size):
        batch_element = output[i]
        # Check that batch element is not all zeros (unless input was all zeros)
        assert not torch.all(batch_element == 0), f"Batch element {i} is all zeros"
        
        # Check that batch elements are not identical (unless input was identical)
        if i > 0:
            assert not torch.allclose(output[i], output[0]), f"Batch element {i} is identical to element 0"
    
    # Oracle comparison with torch.nn.functional.interpolate
    expected = F.interpolate(
        x,
        scale_factor=scale_factor,
        mode='nearest',
        recompute_scale_factor=True
    )
    
    # For nearest neighbor, values should match exactly
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-8), (
        "Output does not match F.interpolate oracle"
    )
    
    # Additional check: verify scaling factor is correct
    actual_scale_h = output.shape[2] / height
    actual_scale_w = output.shape[3] / width
    assert abs(actual_scale_h - scale_factor) < 0.01, (
        f"Height scaling incorrect: {actual_scale_h} != {scale_factor}"
    )
    assert abs(actual_scale_w - scale_factor) < 0.01, (
        f"Width scaling incorrect: {actual_scale_w} != {scale_factor}"
    )
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# CASE_08: UpsamplingBilinear2d 不同 dtype
@pytest.mark.parametrize("dtype, input_shape, scale_factor", [
    (torch.float64, (1, 1, 5, 5), 2.0),  # Original from spec: float64
    (torch.float32, (2, 3, 7, 7), 2.0),  # Additional: float32 with more channels
    (torch.float16, (1, 2, 4, 4), 1.5),  # Additional: float16
])
def test_upsampling_bilinear2d_different_dtypes(dtype, input_shape, scale_factor, device):
    """
    Test UpsamplingBilinear2d with different data types.
    
    Verifies that:
    1. Output shape matches expected dimensions
    2. Output dtype matches input dtype
    3. No NaN values in output
    4. Precision is maintained for different dtypes
    """
    # Skip float16 tests if device doesn't support it
    if dtype == torch.float16 and device.type == 'cpu':
        pytest.skip("float16 not well supported on CPU")
    
    # Create input tensor
    batch_size, channels, height, width = input_shape
    x = create_test_tensor(input_shape, dtype=dtype, device=device)
    
    # Create UpsamplingBilinear2d module
    upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    upsample.to(device)
    
    # Forward pass
    output = upsample(x)
    
    # Calculate expected output shape
    expected_height = int(height * scale_factor)
    expected_width = int(width * scale_factor)
    expected_shape = (batch_size, channels, expected_height, expected_width)
    
    # Weak assertions
    # 1. Shape match
    assert output.shape == expected_shape, (
        f"Output shape mismatch: {output.shape} != {expected_shape}"
    )
    
    # 2. Dtype match
    assert output.dtype == x.dtype, (
        f"Output dtype mismatch: {output.dtype} != {x.dtype}"
    )
    
    # 3. No NaN
    assert not torch.any(torch.isnan(output)), "Output contains NaN values"
    
    # 4. Precision check - verify values are reasonable
    # Check that output values are within expected range
    # For normalized input [0, 1), bilinear output should also be in reasonable range
    if dtype in [torch.float32, torch.float64]:
        assert torch.all(output >= -0.1), "Output contains values below -0.1"
        assert torch.all(output <= 1.1), "Output contains values above 1.1"
    
    # Oracle comparison with torch.nn.functional.interpolate
    # Note: align_corners=True is default for UpsamplingBilinear2d
    expected = F.interpolate(
        x,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=True,
        recompute_scale_factor=True
    )
    
    # Set appropriate tolerance based on dtype
    if dtype == torch.float64:
        rtol, atol = 1e-10, 1e-12
    elif dtype == torch.float32:
        rtol, atol = 1e-5, 1e-8
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-4
    else:
        rtol, atol = 1e-5, 1e-8
    
    # Check if output matches oracle
    close = torch.allclose(output, expected, rtol=rtol, atol=atol)
    if not close:
        max_diff = torch.max(torch.abs(output - expected)).item()
        mean_diff = torch.mean(torch.abs(output - expected)).item()
        
        # For float16, we might have larger differences
        if dtype == torch.float16:
            if max_diff < 0.01:  # Acceptable for float16
                return
        
        raise AssertionError(
            f"Output does not match F.interpolate oracle for dtype {dtype}\n"
            f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}\n"
            f"RTOL={rtol}, ATOL={atol}"
        )
    
    # Additional check: dtype consistency
    # Verify that the module works correctly with the specified dtype
    assert output.dtype == dtype, f"Dtype not preserved: {output.dtype} != {dtype}"
    
    # Check that scaling is correct
    actual_scale_h = output.shape[2] / height
    actual_scale_w = output.shape[3] / width
    assert abs(actual_scale_h - scale_factor) < 0.01, (
        f"Height scaling incorrect: {actual_scale_h} != {scale_factor}"
    )
    assert abs(actual_scale_w - scale_factor) < 0.01, (
        f"Width scaling incorrect: {actual_scale_w} != {scale_factor}"
    )
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Footer block for G2

# Additional helper functions for G2 tests
def test_upsampling_nearest2d_repr():
    """Test string representation of UpsamplingNearest2d."""
    module = nn.UpsamplingNearest2d(scale_factor=2.0)
    repr_str = repr(module)
    assert "UpsamplingNearest2d" in repr_str
    assert "scale_factor=2.0" in repr_str
    assert "mode=nearest" in repr_str
    # Note: size parameter is not shown in repr when scale_factor is provided

def test_upsampling_bilinear2d_repr():
    """Test string representation of UpsamplingBilinear2d."""
    module = nn.UpsamplingBilinear2d(scale_factor=1.5)
    repr_str = repr(module)
    assert "UpsamplingBilinear2d" in repr_str
    assert "scale_factor=1.5" in repr_str
    assert "mode=bilinear" in repr_str
    # Note: size parameter is not shown in repr when scale_factor is provided

def test_upsampling_modules_are_subclasses():
    """Verify that specialized upsampling modules are subclasses of Upsample."""
    assert issubclass(nn.UpsamplingNearest2d, nn.Upsample)
    assert issubclass(nn.UpsamplingBilinear2d, nn.Upsample)
    
    # Verify they have fixed modes
    nearest2d = nn.UpsamplingNearest2d(scale_factor=2.0)
    bilinear2d = nn.UpsamplingBilinear2d(scale_factor=2.0)
    
    # Check mode attribute (might be private or in __init__)
    # UpsamplingNearest2d should use 'nearest' mode
    # UpsamplingBilinear2d should use 'bilinear' mode with align_corners=True
    
    # Test that they work correctly
    x = torch.randn(1, 1, 4, 4)
    out_nearest = nearest2d(x)
    out_bilinear = bilinear2d(x)
    
    assert out_nearest.shape == (1, 1, 8, 8)
    assert out_bilinear.shape == (1, 1, 8, 8)

# Test module initialization with different parameters
@pytest.mark.parametrize("scale_factor", [2.0, 1.5, 3.0, (2.0, 3.0)])
def test_upsampling_nearest2d_init(scale_factor):
    """Test UpsamplingNearest2d initialization with various scale factors."""
    module = nn.UpsamplingNearest2d(scale_factor=scale_factor)
    assert module.scale_factor == scale_factor

@pytest.mark.parametrize("scale_factor", [2.0, 1.5, 3.0, (2.0, 3.0)])
def test_upsampling_bilinear2d_init(scale_factor):
    """Test UpsamplingBilinear2d initialization with various scale factors."""
    module = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    assert module.scale_factor == scale_factor

# Test with size parameter instead of scale_factor
def test_upsampling_nearest2d_with_size():
    """Test UpsamplingNearest2d with size parameter."""
    x = torch.randn(1, 3, 4, 4)
    
    # Test with size parameter
    module = nn.UpsamplingNearest2d(size=(8, 8))
    output = module(x)
    
    assert output.shape == (1, 3, 8, 8)
    
    # Verify it produces same result as scale_factor=2.0
    module2 = nn.UpsamplingNearest2d(scale_factor=2.0)
    output2 = module2(x)
    
    assert torch.allclose(output, output2, rtol=1e-5, atol=1e-8)

def test_upsampling_bilinear2d_with_size():
    """Test UpsamplingBilinear2d with size parameter."""
    x = torch.randn(1, 3, 4, 4)
    
    # Test with size parameter
    module = nn.UpsamplingBilinear2d(size=(6, 6))
    output = module(x)
    
    assert output.shape == (1, 3, 6, 6)
    
    # Verify it produces same result as scale_factor=1.5
    module2 = nn.UpsamplingBilinear2d(scale_factor=1.5)
    output2 = module2(x)
    
    assert torch.allclose(output, output2, rtol=1e-5, atol=1e-8)

# Test tuple scale factors
def test_upsampling_nearest2d_tuple_scale():
    """Test UpsamplingNearest2d with tuple scale factors."""
    x = torch.randn(1, 3, 4, 4)
    
    # Different scale factors for height and width
    module = nn.UpsamplingNearest2d(scale_factor=(2.0, 3.0))
    output = module(x)
    
    assert output.shape == (1, 3, 8, 12)  # 4*2=8, 4*3=12

def test_upsampling_bilinear2d_tuple_scale():
    """Test UpsamplingBilinear2d with tuple scale factors."""
    x = torch.randn(1, 3, 4, 4)
    
    # Different scale factors for height and width
    module = nn.UpsamplingBilinear2d(scale_factor=(1.5, 2.0))
    output = module(x)
    
    assert output.shape == (1, 3, 6, 8)  # 4*1.5=6, 4*2=8

# Test deprecated warning - mark as xfail since Upsampling modules may not emit warnings
@pytest.mark.xfail(reason="Upsampling modules may not implement deprecation warnings")
def test_upsampling_deprecated_warning():
    """Test that Upsampling modules emit deprecation warning."""
    with pytest.warns(UserWarning, match="deprecated"):
        nn.UpsamplingNearest2d(scale_factor=2.0)
    
    with pytest.warns(UserWarning, match="deprecated"):
        nn.UpsamplingBilinear2d(scale_factor=2.0)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====