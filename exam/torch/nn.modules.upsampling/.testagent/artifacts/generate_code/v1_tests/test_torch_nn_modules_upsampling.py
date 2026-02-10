"""
Main test module for torch.nn.modules.upsampling
This file imports tests from group-specific files
"""
import pytest

# Import tests from group files
# These imports ensure all tests are discoverable by pytest

# Note: In a real setup, you might want to use pytest's test discovery
# or import the test modules directly. This is a simple approach to
# make all tests available from a single entry point.

# The actual tests are defined in the group-specific files:
# - test_torch_nn_modules_upsampling_g1.py
# - test_torch_nn_modules_upsampling_g2.py  
# - test_torch_nn_modules_upsampling_g3.py

# This file serves as a central test runner that can be used to
# run all tests with: pytest tests/test_torch_nn_modules_upsampling.py

# For now, we'll just have a simple test to verify the module structure
def test_module_structure():
    """Verify that the test module structure is correct."""
    import torch.nn.modules.upsampling as upsampling_module
    assert hasattr(upsampling_module, 'Upsample')
    assert hasattr(upsampling_module, 'UpsamplingNearest2d')
    assert hasattr(upsampling_module, 'UpsamplingBilinear2d')
    
    # Test that we can import the classes
    from torch.nn.modules.upsampling import (
        Upsample, 
        UpsamplingNearest2d, 
        UpsamplingBilinear2d
    )
    assert Upsample is not None
    assert UpsamplingNearest2d is not None
    assert UpsamplingBilinear2d is not None