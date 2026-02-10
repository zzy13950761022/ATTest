import math
import pytest
import torch
import torch.nn as nn
from torch.nn.modules.fold import Fold, Unfold

# ==== BLOCK:HEADER START ====
"""
Test module for torch.nn.modules.fold.Fold
G1: Fold类核心功能
"""
import math
import pytest
import torch
import torch.nn as nn
from torch.nn.modules.fold import Fold, Unfold

# Set random seed for reproducibility
torch.manual_seed(42)

# Helper functions
def compute_fold_output_shape(batch_size, channels, output_size, kernel_size, stride, padding, dilation):
    """Compute expected output shape for Fold operation"""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Fold output shape is (N, C, output_size[0], output_size[1])
    return (batch_size, channels, output_size[0], output_size[1])

def create_fold_input(batch_size, channels, output_size, kernel_size, stride, padding, dilation, dtype=torch.float32, device='cpu'):
    """Create valid input tensor for Fold operation"""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Compute number of blocks
    # Formula: L = ∏_{i=1}^2 ⌊(output_size[i] + 2*padding[i] - dilation[i]*(kernel_size[i]-1) - 1)/stride[i] + 1⌋
    blocks_h = math.floor((output_size[0] + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    blocks_w = math.floor((output_size[1] + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
    total_blocks = blocks_h * blocks_w
    
    # Input shape for Fold: (N, C * ∏(kernel_size), L)
    input_channels = channels * kernel_size[0] * kernel_size[1]
    input_tensor = torch.randn(batch_size, input_channels, total_blocks, dtype=dtype, device=device)
    return input_tensor
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize(
    "output_size,kernel_size,stride,padding,dilation,batch_size,channels,dtype,device",
    [
        # Base case from test plan
        (4, 2, 1, 0, 1, 1, 3, torch.float32, 'cpu'),
        # Parameter extensions
        (4, 2, 1, 0, 1, 4, 8, torch.float64, 'cpu'),
    ]
)
def test_fold_basic_int_params(output_size, kernel_size, stride, padding, dilation, batch_size, channels, dtype, device):
    """
    TC-01: Fold基本功能_int参数
    Test Fold with integer parameters (scalar values applied to all dimensions)
    """
    # Create Fold module
    fold = Fold(
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    
    # Create valid input tensor
    input_tensor = create_fold_input(
        batch_size=batch_size,
        channels=channels,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        device=device
    )
    
    # Forward pass
    output = fold(input_tensor)
    
    # Weak assertions
    # 1. Shape match
    expected_shape = compute_fold_output_shape(
        batch_size, channels, output_size, kernel_size, stride, padding, dilation
    )
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # 2. Dtype match
    assert output.dtype == dtype, f"Expected dtype {dtype}, got {output.dtype}"
    
    # 3. Finite values
    assert torch.isfinite(output).all(), "Output contains non-finite values (inf or nan)"
    
    # 4. No NaN/Inf
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    # Additional basic checks
    assert output.device.type == device, f"Expected device {device}, got {output.device.type}"
    
    # Verify that output has correct spatial dimensions
    if isinstance(output_size, int):
        assert output.size(2) == output_size, f"Expected height {output_size}, got {output.size(2)}"
        assert output.size(3) == output_size, f"Expected width {output_size}, got {output.size(3)}"
    else:
        assert output.size(2) == output_size[0], f"Expected height {output_size[0]}, got {output.size(2)}"
        assert output.size(3) == output_size[1], f"Expected width {output_size[1]}, got {output.size(3)}"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize(
    "output_size,kernel_size,stride,padding,dilation,batch_size,channels,dtype,device",
    [
        # Base case from test plan
        ([4, 5], [2, 3], [1, 1], [0, 0], [1, 1], 2, 3, torch.float32, 'cpu'),
        # Parameter extensions
        ([4, 5], [2, 3], [2, 2], [1, 1], [2, 2], 2, 3, torch.float32, 'cpu'),
    ]
)
def test_fold_basic_tuple_params(output_size, kernel_size, stride, padding, dilation, batch_size, channels, dtype, device):
    """
    TC-02: Fold基本功能_tuple参数
    Test Fold with tuple parameters (different values for height and width dimensions)
    """
    # Create Fold module
    fold = Fold(
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    
    # Create valid input tensor
    input_tensor = create_fold_input(
        batch_size=batch_size,
        channels=channels,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        device=device
    )
    
    # Forward pass
    output = fold(input_tensor)
    
    # Weak assertions
    # 1. Shape match
    expected_shape = compute_fold_output_shape(
        batch_size, channels, output_size, kernel_size, stride, padding, dilation
    )
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # 2. Dtype match
    assert output.dtype == dtype, f"Expected dtype {dtype}, got {output.dtype}"
    
    # 3. Finite values
    assert torch.isfinite(output).all(), "Output contains non-finite values (inf or nan)"
    
    # 4. No NaN/Inf
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    # Additional basic checks
    assert output.device.type == device, f"Expected device {device}, got {output.device.type}"
    
    # Verify that output has correct spatial dimensions
    assert output.size(2) == output_size[0], f"Expected height {output_size[0]}, got {output.size(2)}"
    assert output.size(3) == output_size[1], f"Expected width {output_size[1]}, got {output.size(3)}"
    
    # Verify parameter handling
    # Check that module stores parameters correctly
    # Compare as lists to handle both list and tuple storage
    assert list(fold.output_size) == list(output_size), \
        f"Module output_size mismatch: expected {output_size}, got {fold.output_size}"
    assert list(fold.kernel_size) == list(kernel_size), \
        f"Module kernel_size mismatch: expected {kernel_size}, got {fold.kernel_size}"
    assert list(fold.stride) == list(stride), \
        f"Module stride mismatch: expected {stride}, got {fold.stride}"
    assert list(fold.padding) == list(padding), \
        f"Module padding mismatch: expected {padding}, got {fold.padding}"
    assert list(fold.dilation) == list(dilation), \
        f"Module dilation mismatch: expected {dilation}, got {fold.dilation}"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize(
    "output_size,kernel_size,stride,padding,dilation,batch_size,channels,dtype,device",
    [
        # Base case from test plan - minimal valid configuration
        ([2, 2], [2, 2], [1, 1], [0, 0], [1, 1], 1, 1, torch.float32, 'cpu'),
        # Additional edge case - single element output with padding
        ([1, 1], [1, 1], [1, 1], [0, 0], [1, 1], 1, 1, torch.float32, 'cpu'),
    ]
)
def test_fold_edge_conditions(output_size, kernel_size, stride, padding, dilation, batch_size, channels, dtype, device):
    """
    TC-03: Fold边界条件
    Test Fold with edge/boundary conditions (minimal valid input size)
    """
    # Create Fold module
    fold = Fold(
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    
    # Create valid input tensor with minimal size
    input_tensor = create_fold_input(
        batch_size=batch_size,
        channels=channels,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        device=device
    )
    
    # Forward pass
    output = fold(input_tensor)
    
    # Weak assertions
    # 1. Shape match
    expected_shape = compute_fold_output_shape(
        batch_size, channels, output_size, kernel_size, stride, padding, dilation
    )
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # 2. Dtype match
    assert output.dtype == dtype, f"Expected dtype {dtype}, got {output.dtype}"
    
    # 3. Finite values
    assert torch.isfinite(output).all(), "Output contains non-finite values (inf or nan)"
    
    # 4. No NaN/Inf
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    # Additional basic checks
    assert output.device.type == device, f"Expected device {device}, got {output.device.type}"
    
    # Verify parameter handling
    # Check that module stores parameters correctly
    # Compare as lists to handle both list and tuple storage
    assert list(fold.output_size) == list(output_size), \
        f"Module output_size mismatch: expected {output_size}, got {fold.output_size}"
    assert list(fold.kernel_size) == list(kernel_size), \
        f"Module kernel_size mismatch: expected {kernel_size}, got {fold.kernel_size}"
    assert list(fold.stride) == list(stride), \
        f"Module stride mismatch: expected {stride}, got {fold.stride}"
    assert list(fold.padding) == list(padding), \
        f"Module padding mismatch: expected {padding}, got {fold.padding}"
    assert list(fold.dilation) == list(dilation), \
        f"Module dilation mismatch: expected {dilation}, got {fold.dilation}"
    
    # Verify that output has correct spatial dimensions
    assert output.size(2) == output_size[0], f"Expected height {output_size[0]}, got {output.size(2)}"
    assert output.size(3) == output_size[1], f"Expected width {output_size[1]}, got {output.size(3)}"
    
    # Verify input shape calculation
    # For 2x2 output with 2x2 kernel, stride 1, padding 0, dilation 1:
    # blocks_h = floor((2 + 2*0 - 1*(2-1) - 1)/1 + 1) = floor((2 - 1 - 1)/1 + 1) = floor(0 + 1) = 1
    # blocks_w = same = 1
    # total_blocks = 1 * 1 = 1
    # input_channels = channels * kernel_size[0] * kernel_size[1] = 1 * 2 * 2 = 4
    # Expected input shape: (1, 4, 1)
    blocks_h = math.floor((output_size[0] + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    blocks_w = math.floor((output_size[1] + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
    expected_input_blocks = blocks_h * blocks_w
    expected_input_channels = channels * kernel_size[0] * kernel_size[1]
    assert input_tensor.size(1) == expected_input_channels, \
        f"Expected input channels {expected_input_channels}, got {input_tensor.size(1)}"
    assert input_tensor.size(2) == expected_input_blocks, \
        f"Expected {expected_input_blocks} blocks in input, got {input_tensor.size(2)}"
    
    # Verify that output values are reasonable
    # Since input is random, we just check that output has same magnitude as input
    # The fold operation sums overlapping contributions, so output values may be larger
    # We'll just check they're not all zeros (unless input was zero)
    assert not torch.allclose(output, torch.zeros_like(output), rtol=1e-5, atol=1e-8), \
        "Output is all zeros (unexpected for random input)"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
@pytest.mark.parametrize(
    "output_size,kernel_size,stride,padding,dilation,batch_size,channels,dtype,device,expect_error",
    [
        # Error case from test plan - kernel_size > output_size should raise an error
        ([2, 2], [3, 3], [1, 1], [0, 0], [1, 1], 1, 1, torch.float32, 'cpu', True),
        # Additional valid case to cover else branch
        ([4, 4], [2, 2], [1, 1], [0, 0], [1, 1], 1, 1, torch.float32, 'cpu', False),
    ]
)
def test_fold_error_handling(output_size, kernel_size, stride, padding, dilation, batch_size, channels, dtype, device, expect_error):
    """
    TC-04: Fold错误处理
    Test Fold error handling for invalid parameters
    """
    if expect_error:
        # Test that invalid parameters raise appropriate exceptions
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            # Create Fold module with invalid parameters
            fold = Fold(
                output_size=output_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
            
            # Try to create input tensor (might fail earlier)
            # For kernel_size > output_size, the input creation formula might produce
            # negative or zero blocks, which should also raise an error
            try:
                input_tensor = create_fold_input(
                    batch_size=batch_size,
                    channels=channels,
                    output_size=output_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    dtype=dtype,
                    device=device
                )
                
                # Try forward pass (should fail if module creation succeeded)
                output = fold(input_tensor)
            except (ValueError, RuntimeError) as inner_exc:
                # Input creation or forward pass failed as expected
                raise inner_exc
        
        # Verify error message contains relevant information
        error_msg = str(exc_info.value).lower()
        # Check for shape or size related error
        # The error could be about kernel size, output size, or block calculation
        assert any(keyword in error_msg for keyword in ['kernel', 'size', 'output', 'shape', 'block', 'invalid', 'positive']), \
            f"Error message should mention kernel/size/shape issue, got: {error_msg}"
        
        # Additional check: verify the specific condition
        # kernel_size (3,3) > output_size (2,2) should definitely cause an error
        # because you can't have a 3x3 kernel sliding over a 2x2 output with padding=0
        kernel_h, kernel_w = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        output_h, output_w = output_size if isinstance(output_size, (list, tuple)) else (output_size, output_size)
        
        if kernel_h > output_h or kernel_w > output_w:
            # This is the expected condition for this test case
            assert True, "Correctly detected kernel_size > output_size condition"
    else:
        # This branch is for future extensions if needed
        # Create Fold module with valid parameters
        fold = Fold(
            output_size=output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        
        # Create valid input tensor
        input_tensor = create_fold_input(
            batch_size=batch_size,
            channels=channels,
            output_size=output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dtype=dtype,
            device=device
        )
        
        # Forward pass
        output = fold(input_tensor)
        
        # Basic assertions
        assert output.shape[0] == batch_size
        assert output.shape[1] == channels
        assert output.dtype == dtype
        assert output.device.type == device
        
        # Additional weak assertions
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:FOOTER START ====
# Footer block
# ==== BLOCK:FOOTER END ====