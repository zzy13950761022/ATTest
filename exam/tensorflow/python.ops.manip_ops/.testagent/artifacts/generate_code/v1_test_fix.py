#!/usr/bin/env python3
"""Quick test to verify the fixes for CASE_04 and CASE_05."""

import numpy as np
import tensorflow as tf
import pytest
from tensorflow.python.ops import manip_ops

# Test the fixed create_test_tensor function
print("Testing create_test_tensor function...")

# Test scalar case
scalar_tensor = tf.constant(5.0, dtype=tf.float32)
print(f"Scalar tensor: {scalar_tensor}, shape: {scalar_tensor.shape}, dtype: {scalar_tensor.dtype}")

# Test empty tensor case  
empty_tensor = tf.constant([], dtype=tf.int32)
print(f"Empty tensor: {empty_tensor}, shape: {empty_tensor.shape}, dtype: {empty_tensor.dtype}")

# Test roll on scalar
print("\nTesting roll on scalar tensor...")
scalar_result = manip_ops.roll(scalar_tensor, shift=0, axis=0)
print(f"Scalar roll result: {scalar_result}, shape: {scalar_result.shape}")

# Test roll on empty tensor
print("\nTesting roll on empty tensor...")
empty_result = manip_ops.roll(empty_tensor, shift=0, axis=0)
print(f"Empty roll result: {empty_result}, shape: {empty_result.shape}")

# Test invalid shift type
print("\nTesting invalid shift type...")
try:
    invalid_result = manip_ops.roll(scalar_tensor, shift="invalid", axis=0)
    print("ERROR: Should have raised exception!")
except tf.errors.InvalidArgumentError as e:
    print(f"✓ Correctly raised InvalidArgumentError: {e}")
except Exception as e:
    print(f"✗ Wrong exception type: {type(e).__name__}: {e}")

# Test shape mismatch
print("\nTesting shape mismatch...")
try:
    mismatch_result = manip_ops.roll(scalar_tensor, shift=[1, 2], axis=[0])
    print("ERROR: Should have raised exception!")
except (ValueError, tf.errors.InvalidArgumentError) as e:
    print(f"✓ Correctly raised exception: {type(e).__name__}: {e}")
except Exception as e:
    print(f"✗ Wrong exception type: {type(e).__name__}: {e}")

print("\nAll tests completed!")