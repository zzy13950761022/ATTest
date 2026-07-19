import pytest
import sys

# Run the tests
exit_code = pytest.main(["-xvs", "tests/test_tensorflow_python_ops_manip_ops.py"])
sys.exit(exit_code)