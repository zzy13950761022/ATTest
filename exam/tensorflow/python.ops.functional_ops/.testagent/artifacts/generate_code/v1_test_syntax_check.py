#!/usr/bin/env python3
"""Quick syntax check for the fixed test file."""

import ast
import sys

def check_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        print(f"✓ Syntax check passed for {filepath}")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {filepath}:")
        print(f"  Line {e.lineno}, Column {e.offset}: {e.msg}")
        print(f"  Text: {e.text}")
        return False
    except Exception as e:
        print(f"✗ Error reading {filepath}: {e}")
        return False

if __name__ == "__main__":
    test_file = "tests/test_tensorflow_python_ops_functional_ops.py"
    success = check_syntax(test_file)
    sys.exit(0 if success else 1)