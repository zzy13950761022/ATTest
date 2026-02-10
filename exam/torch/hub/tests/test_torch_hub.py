"""
Main test file for torch.hub module.
This file imports tests from group-specific files.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tests from group files
from test_torch_hub_load import *
from test_torch_hub_trust import *

if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))