#!/usr/bin/env python3
"""Test import of test files."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test G1 import
    print("Testing G1 import...")
    import tests.test_tensorflow_python_data_experimental_ops_batching_g1 as g1
    print("✓ G1 import successful")
    
    # Test G2 import
    print("Testing G2 import...")
    import tests.test_tensorflow_python_data_experimental_ops_batching_g2 as g2
    print("✓ G2 import successful")
    
    # Test main import
    print("Testing main import...")
    import tests.test_tensorflow_python_data_experimental_ops_batching as main
    print("✓ Main import successful")
    
    # Check test functions
    print("\nTest functions in G1:")
    for name in dir(g1):
        if name.startswith('test_'):
            print(f"  - {name}")
    
    print("\nTest functions in G2:")
    for name in dir(g2):
        if name.startswith('test_'):
            print(f"  - {name}")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)