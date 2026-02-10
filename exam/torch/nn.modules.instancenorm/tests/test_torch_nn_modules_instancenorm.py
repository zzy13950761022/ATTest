"""
Main test file for torch.nn.modules.instancenorm.
Tests are organized into group files for better modularity.
"""

import sys
import os

# Add current directory to path to import group tests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all test functions from group files
# This makes them discoverable when running this file directly
try:
    from test_torch_nn_modules_instancenorm_g1 import (
        test_instance_norm_basic_forward,
        test_instance_norm_affine_parameters,
        test_lazy_instance_norm_inference,
        test_instance_norm_track_running_stats,
        test_instance_norm_invalid_parameters,
        test_instance_norm_input_dimension_validation,
        test_instance_norm_channel_mismatch,
        set_random_seed,
        assert_tensor_properties
    )
except ImportError as e:
    print(f"Warning: Could not import G1 tests: {e}")

# Import G2 tests
try:
    from test_torch_nn_modules_instancenorm_g2 import (
        test_lazy_instance_norm_inference,
        test_instance_norm_track_running_stats,
        test_lazy_instance_norm_parameter_validation,
        test_lazy_instance_norm_dimension_validation,
        test_lazy_instance_norm_affine_parameters,
        set_random_seed,
        assert_tensor_properties,
        create_lazy_norm_layer
    )
except ImportError as e:
    print(f"Warning: Could not import G2 tests: {e}")