#!/usr/bin/env python3
"""
Fix the test file by removing class references and making functions standalone.
"""
import re

def fix_test_file(input_file, output_file):
    """Fix test file by removing class references."""
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Remove self. references
    content = content.replace('self.create_test_tensor', 'create_test_tensor')
    content = content.replace('self.numpy_roll_reference', 'numpy_roll_reference')
    
    # Remove class method indentation and self parameter
    # Fix test function definitions
    content = re.sub(r'    def test_', 'def test_', content)
    content = re.sub(r'\(self, ', '(', content)
    content = re.sub(r', self\)', ')', content)
    content = re.sub(r'\(self\)', '()', content)
    
    # Remove class definition if it exists
    content = re.sub(r'class TestRollFunction:.*?"""Test class.*?""".*?\n', '', content, flags=re.DOTALL)
    
    # Remove @staticmethod decorators
    content = content.replace('    @staticmethod\n', '')
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Fixed file saved to {output_file}")

if __name__ == "__main__":
    fix_test_file(
        'tests/test_tensorflow_python_ops_manip_ops.py',
        'tests/test_tensorflow_python_ops_manip_ops_fixed.py'
    )