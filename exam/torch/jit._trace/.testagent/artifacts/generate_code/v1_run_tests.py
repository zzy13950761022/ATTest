#!/usr/bin/env python3
"""Simple script to run the generated tests."""

import subprocess
import sys
import os

def run_tests():
    """Run all test files."""
    test_files = [
        "tests/test_torch_jit_trace_g1.py",
        "tests/test_torch_jit_trace_g2.py", 
        "tests/test_torch_jit_trace_g3.py",
        "tests/test_torch_jit_trace_g4.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n{'='*60}")
            print(f"Running {test_file}")
            print('='*60)
            
            try:
                # Run the test file directly
                result = subprocess.run(
                    [sys.executable, test_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print(f"✓ {test_file} passed")
                else:
                    print(f"✗ {test_file} failed with return code {result.returncode}")
                    print(f"Stdout:\n{result.stdout}")
                    print(f"Stderr:\n{result.stderr}")
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                print(f"✗ {test_file} timed out after 30 seconds")
                all_passed = False
            except Exception as e:
                print(f"✗ Error running {test_file}: {e}")
                all_passed = False
        else:
            print(f"✗ Test file not found: {test_file}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print('='*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(run_tests())