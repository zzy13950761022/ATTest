#!/usr/bin/env python3
"""
Verify that the test file is correctly structured.
"""
import ast
import sys

def verify_test_file(filepath):
    """Verify the structure of a test file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = ['import numpy', 'import tensorflow', 'import pytest']
        for imp in required_imports:
            if imp not in content:
                print(f"❌ Missing import: {imp}")
                return False
        
        # Check for BLOCK markers
        block_markers = [
            'BLOCK:HEADER START',
            'BLOCK:HEADER END',
            'BLOCK:CASE_01 START',
            'BLOCK:CASE_01 END',
            'BLOCK:CASE_02 START',
            'BLOCK:CASE_02 END',
            'BLOCK:CASE_03 START',
            'BLOCK:CASE_03 END',
            'BLOCK:CASE_04 START',
            'BLOCK:CASE_04 END',
            'BLOCK:CASE_05 START',
            'BLOCK:CASE_05 END',
            'BLOCK:FOOTER START',
            'BLOCK:FOOTER END'
        ]
        
        for marker in block_markers:
            if marker not in content:
                print(f"❌ Missing block marker: {marker}")
                return False
        
        # Parse the AST to check for test functions
        tree = ast.parse(content)
        test_functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')]
        
        expected_tests = ['test_1d_tensor_basic_roll', 'test_multi_axis_roll', 'test_same_axis_cumulative_roll']
        
        for test_name in expected_tests:
            if not any(func.name == test_name for func in test_functions):
                print(f"❌ Missing test function: {test_name}")
                return False
        
        print(f"✅ Test file {filepath} is correctly structured!")
        print(f"   Found {len(test_functions)} test functions")
        print(f"   All required block markers present")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying file: {e}")
        return False

if __name__ == "__main__":
    test_files = [
        'test_tensorflow_python_ops_manip_ops_fixed.py',
        'test_roll_final.py',
        'test_tensorflow_python_ops_manip_ops_correct.py'
    ]
    
    all_good = True
    for test_file in test_files:
        print(f"\nVerifying {test_file}...")
        if not verify_test_file(f'tests/{test_file}'):
            all_good = False
    
    if all_good:
        print("\n✅ All test files are correctly structured!")
        sys.exit(0)
    else:
        print("\n❌ Some test files have issues")
        sys.exit(1)