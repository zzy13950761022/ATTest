#!/usr/bin/env python3
"""Test to verify mock path issue"""
import sys
from unittest import mock

# Test 1: Try to mock tensorflow.python.saved_model.save.save_and_return_nodes
print("Test 1: Mocking tensorflow.python.saved_model.save.save_and_return_nodes")
try:
    with mock.patch('tensorflow.python.saved_model.save.save_and_return_nodes') as mock_save:
        print("  SUCCESS: Mock created successfully")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Test 2: Try to mock tensorflow.saved_model.save
print("\nTest 2: Mocking tensorflow.saved_model.save")
try:
    with mock.patch('tensorflow.saved_model.save') as mock_save:
        print("  SUCCESS: Mock created successfully")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Test 3: Check if we can import tensorflow.python.saved_model.save directly
print("\nTest 3: Importing tensorflow.python.saved_model.save directly")
try:
    import importlib
    module = importlib.import_module('tensorflow.python.saved_model.save')
    print(f"  SUCCESS: Module imported: {module}")
    if hasattr(module, 'save_and_return_nodes'):
        print("  SUCCESS: save_and_return_nodes function exists")
    else:
        print("  WARNING: save_and_return_nodes function not found")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\nConclusion: The issue is that tensorflow.python is not accessible via tensorflow.python,")
print("but the module exists and can be imported directly. Mock.patch tries to access")
print("tensorflow.python via tensorflow module, which fails.")