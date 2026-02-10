#!/usr/bin/env python
"""Run tests for tensorflow.python.data.experimental.ops.prefetching_ops"""

import subprocess
import sys
import os

def run_tests():
    """Run pytest on test files."""
    print("Running tests for tensorflow.python.data.experimental.ops.prefetching_ops")
    print("=" * 70)
    
    # Run tests for G1 group
    print("\nRunning G1 tests (prefetch_to_device核心功能)...")
    result_g1 = subprocess.run(
        [sys.executable, "-m", "pytest", 
         "tests/test_tensorflow_python_data_experimental_ops_prefetching_ops_g1.py",
         "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print("G1 Test Output:")
    print(result_g1.stdout)
    if result_g1.stderr:
        print("G1 Test Errors:")
        print(result_g1.stderr)
    
    # Run tests for G2 group  
    print("\nRunning G2 tests (copy_to_device与设备间传输)...")
    result_g2 = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_tensorflow_python_data_experimental_ops_prefetching_ops_g2.py",
         "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print("G2 Test Output:")
    print(result_g2.stdout)
    if result_g2.stderr:
        print("G2 Test Errors:")
        print(result_g2.stderr)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    # Parse G1 results
    g1_passed = result_g1.stdout.count("PASSED")
    g1_failed = result_g1.stdout.count("FAILED")
    g1_skipped = result_g1.stdout.count("SKIPPED")
    g1_errors = result_g1.stdout.count("ERROR")
    
    # Parse G2 results
    g2_passed = result_g2.stdout.count("PASSED")
    g2_failed = result_g2.stdout.count("FAILED")
    g2_skipped = result_g2.stdout.count("SKIPPED")
    g2_errors = result_g2.stdout.count("ERROR")
    
    total_passed = g1_passed + g2_passed
    total_failed = g1_failed + g2_failed
    total_skipped = g1_skipped + g2_skipped
    total_errors = g1_errors + g2_errors
    
    print(f"G1 Tests: {g1_passed} passed, {g1_failed} failed, {g1_skipped} skipped, {g1_errors} errors")
    print(f"G2 Tests: {g2_passed} passed, {g2_failed} failed, {g2_skipped} skipped, {g2_errors} errors")
    print(f"Total:    {total_passed} passed, {total_failed} failed, {total_skipped} skipped, {total_errors} errors")
    
    # Return code
    if total_failed > 0 or total_errors > 0:
        print("\n❌ Some tests failed!")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(run_tests())