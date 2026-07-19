"""
Main test file for tensorflow.python.framework.config module.
This file imports tests from group-specific files.
"""
# Import G1 tests
from .test_tensorflow_python_framework_config_g1 import (
    test_device_list_query_basic_functionality,
    test_thread_configuration_set_and_query_consistency,
    test_invalid_device_type_exception_handling,
)

# G2 tests will be imported when that group becomes active
# from .test_tensorflow_python_framework_config_g2 import (
#     test_memory_growth_configuration_basic_functionality,
#     test_tensorfloat_32_switch_state_control,
# )

# This allows pytest to discover all tests when running this file directly
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])