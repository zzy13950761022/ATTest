#!/usr/bin/env python3
"""Quick test to verify fixes."""

import pytest
import sys

# Run a subset of tests to verify fixes
if __name__ == "__main__":
    # Test specific functions that were fixed
    test_args = [
        "tests/test_tensorflow_python_training_input.py",
        "-v",
        "-k", "test_batch_invalid_batch_size or test_shuffle_batch_invalid_min_after_dequeue or test_dynamic_padding_without_shapes_raises_error",
        "--tb=short"
    ]
    
    print("Running tests to verify fixes...")
    exit_code = pytest.main(test_args)
    sys.exit(exit_code)