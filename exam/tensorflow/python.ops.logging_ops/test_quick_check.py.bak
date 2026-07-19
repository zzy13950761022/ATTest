#!/usr/bin/env python3
"""Quick test to verify the fixes."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import tensorflow as tf
import numpy as np

# Test CASE_01 fix
print("Testing CASE_01 fix...")
from tensorflow.python.ops import logging_ops

# Test basic print_v2 functionality
result = logging_ops.print_v2("test", tf.constant([1, 2, 3]), output_stream=sys.stdout)
print(f"print_v2 returned: {result} (should be None in eager mode)")

# Test CASE_02 fix
print("\nTesting CASE_02 fix...")
# Create a test tensor
shape = [2, 32, 32, 3]
tensor = tf.constant(np.random.randint(0, 255, shape).astype(np.uint8))
tag = "test_image"

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    result = logging_ops.image_summary(tag=tag, tensor=tensor, max_images=3)
    print(f"image_summary returned tensor with dtype: {result.dtype}, shape: {result.shape}")

# Test CASE_04 fix
print("\nTesting CASE_04 fix...")
tags = ["tag1", "tag2", "tag3"]
values = [tf.constant(1.0), tf.constant(2.0), tf.constant(3.0)]
tags_tensor = tf.constant(tags, dtype=tf.string)
values_tensor = tf.stack([tf.cast(v, tf.float32) for v in values])

print(f"Tags shape: {tags_tensor.shape}, Values shape: {values_tensor.shape}")
assert tags_tensor.shape == values_tensor.shape, "Shapes must match"

result = logging_ops.scalar_summary(tags=tags_tensor, values=values_tensor)
print(f"scalar_summary returned tensor with dtype: {result.dtype}, shape: {result.shape}")

print("\nAll quick tests passed!")