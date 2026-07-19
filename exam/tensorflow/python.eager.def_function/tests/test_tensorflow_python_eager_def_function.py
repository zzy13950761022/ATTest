"""
Main test file for tensorflow.python.eager.def_function.
This file imports and runs tests from both G1 and G2 groups.
"""
import pytest

# Import test classes from group files
from tests.test_tensorflow_python_eager_def_function_g1 import TestDefFunctionG1
from tests.test_tensorflow_python_eager_def_function_g2 import TestDefFunctionG2

# Re-export test classes for pytest discovery
__all__ = ['TestDefFunctionG1', 'TestDefFunctionG2']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])