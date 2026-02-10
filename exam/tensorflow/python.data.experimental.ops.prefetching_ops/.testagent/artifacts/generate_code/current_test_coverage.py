#!/usr/bin/env python
"""Run tests with coverage for tensorflow.python.data.experimental.ops.prefetching_ops"""

import subprocess
import sys
import os

def run_coverage():
    """Run pytest with coverage."""
    print("Running tests with coverage analysis...")
    print("=" * 70)
    
    # Run coverage on both test files
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_tensorflow_python_data_experimental_ops_prefetching_ops_g1.py",
         "tests/test_tensorflow_python_data_experimental_ops_prefetching_ops_g2.py",
         "-v", "--tb=short",
         "--cov=.", "--cov-report=term-missing"],
        capture_output=True,
        text=True
    )
    
    print("Test Output:")
    print(result.stdout)
    if result.stderr:
        print("Test Errors:")
        print(result.stderr)
    
    # Check for coverage issues
    if "Missing" in result.stdout:
        print("\n" + "=" * 70)
        print("COVERAGE ANALYSIS")
        print("=" * 70)
        
        # Extract coverage information
        lines = result.stdout.split('\n')
        in_coverage_section = False
        missing_lines = []
        
        for line in lines:
            if "Missing" in line and "Cover" in line:
                in_coverage_section = True
                continue
            if in_coverage_section and line.strip() and "---" not in line:
                missing_lines.append(line.strip())
            elif in_coverage_section and not line.strip():
                break
        
        if missing_lines:
            print("Lines with missing coverage:")
            for line in missing_lines:
                print(f"  {line}")
        else:
            print("No missing coverage lines found.")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_coverage())