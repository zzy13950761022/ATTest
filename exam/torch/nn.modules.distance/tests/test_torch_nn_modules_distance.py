"""
Main test file for torch.nn.modules.distance
This file imports tests from the group-specific test files.
"""
# Import tests from group-specific files
# This allows running all tests with a single command

# Note: Tests are organized in separate files by group:
# - tests/test_torch_nn_modules_distance_pairwise.py (G1: PairwiseDistance)
# - tests/test_torch_nn_modules_distance_cosine.py (G2: CosineSimilarity)

if __name__ == "__main__":
    import pytest
    import sys
    
    # Run all tests in the tests directory
    sys.exit(pytest.main(["-v", "tests/"]))