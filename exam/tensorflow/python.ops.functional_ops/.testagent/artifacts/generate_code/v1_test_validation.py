#!/usr/bin/env python3
"""Validate the generated test file."""

import ast
import sys

def validate_python_syntax(filepath):
    """Validate Python syntax of a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        ast.parse(content)
        print(f"✓ Syntax validation passed for {filepath}")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {filepath}: {e}")
        print(f"  Line {e.lineno}, column {e.offset}: {e.text}")
        return False
    except Exception as e:
        print(f"✗ Error reading {filepath}: {e}")
        return False

def check_block_markers(filepath):
    """Check that all BLOCK markers are properly formatted."""
    required_blocks = ['HEADER', 'CASE_01', 'CASE_02', 'CASE_03', 'CASE_04', 'CASE_05', 'FOOTER']
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for block in required_blocks:
            start_marker = f'# ==== BLOCK:{block} START ===='
            end_marker = f'# ==== BLOCK:{block} END ===='
            
            if start_marker not in content:
                print(f"✗ Missing START marker for BLOCK:{block}")
                return False
            if end_marker not in content:
                print(f"✗ Missing END marker for BLOCK:{block}")
                return False
            
            # Check that END comes after START
            start_pos = content.find(start_marker)
            end_pos = content.find(end_marker)
            if end_pos <= start_pos:
                print(f"✗ END marker before START marker for BLOCK:{block}")
                return False
        
        print("✓ All BLOCK markers are properly formatted")
        return True
    except Exception as e:
        print(f"✗ Error checking block markers: {e}")
        return False

def check_imports(filepath):
    """Check that required imports are present."""
    required_imports = [
        'import numpy',
        'import tensorflow',
        'import pytest',
        'from unittest import mock'
    ]
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for imp in required_imports:
            if imp not in content:
                print(f"✗ Missing import: {imp}")
                return False
        
        print("✓ All required imports are present")
        return True
    except Exception as e:
        print(f"✗ Error checking imports: {e}")
        return False

def main():
    """Main validation function."""
    test_file = 'tests/test_tensorflow_python_ops_functional_ops.py'
    
    print(f"Validating {test_file}...")
    print("-" * 50)
    
    results = []
    
    # Run validations
    results.append(validate_python_syntax(test_file))
    results.append(check_block_markers(test_file))
    results.append(check_imports(test_file))
    
    print("-" * 50)
    
    if all(results):
        print("✓ All validations passed!")
        return 0
    else:
        print("✗ Some validations failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())