#!/usr/bin/env python3
"""Simple validation script to test the implemented test cases"""
import subprocess
import sys

def run_tests():
    """Run pytest on the test files"""
    print("Running tests for torch.nn.modules.distance...")
    
    # Run tests for pairwise distance
    print("\n=== Testing PairwiseDistance ===")
    result_pairwise = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_torch_nn_modules_distance_pairwise.py", "-v"],
        capture_output=True,
        text=True
    )
    
    print("STDOUT:")
    print(result_pairwise.stdout)
    if result_pairwise.stderr:
        print("STDERR:")
        print(result_pairwise.stderr)
    
    # Run tests for cosine similarity
    print("\n=== Testing CosineSimilarity ===")
    result_cosine = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_torch_nn_modules_distance_cosine.py", "-v"],
        capture_output=True,
        text=True
    )
    
    print("STDOUT:")
    print(result_cosine.stdout)
    if result_cosine.stderr:
        print("STDERR:")
        print(result_cosine.stderr)
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"PairwiseDistance tests return code: {result_pairwise.returncode}")
    print(f"CosineSimilarity tests return code: {result_cosine.returncode}")
    
    if result_pairwise.returncode == 0 and result_cosine.returncode == 0:
        print("All tests passed!")
        return True
    else:
        print("Some tests failed.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)