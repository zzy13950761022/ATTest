import math
import pytest
import torch
import torch.nn as nn
from torch.nn.modules.instancenorm import (
    InstanceNorm1d, InstanceNorm2d, InstanceNorm3d,
    LazyInstanceNorm1d, LazyInstanceNorm2d, LazyInstanceNorm3d
)

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G2 group
@pytest.fixture(scope="function", autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    yield

def assert_tensor_properties(tensor, expected_shape=None, expected_dtype=None, 
                           allow_nan_inf=False, name=""):
    """Assert tensor properties with descriptive error messages."""
    if expected_shape is not None:
        assert tensor.shape == expected_shape, \
            f"{name}: Expected shape {expected_shape}, got {tensor.shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, \
            f"{name}: Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    if not allow_nan_inf:
        assert torch.isfinite(tensor).all(), \
            f"{name}: Tensor contains NaN or Inf values"
    
    return True

def create_lazy_norm_layer(norm_class, affine=False, track_running_stats=False, dtype=torch.float32):
    """Create lazy instance normalization layer."""
    if norm_class == "LazyInstanceNorm1d":
        return LazyInstanceNorm1d(
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype
        )
    elif norm_class == "LazyInstanceNorm2d":
        return LazyInstanceNorm2d(
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype
        )
    elif norm_class == "LazyInstanceNorm3d":
        return LazyInstanceNorm3d(
            affine=affine,
            track_running_stats=track_running_stats,
            dtype=dtype
        )
    else:
        raise ValueError(f"Unsupported lazy norm_class: {norm_class}")
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: LazyInstanceNorm自动推断
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: track_running_stats功能
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: 无批次输入处理 (DEFERRED - placeholder)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# DEFERRED - placeholder
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# DEFERRED - placeholder
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# DEFERRED - placeholder
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Additional test cases and helper functions for G2 group
# ==== BLOCK:FOOTER END ====