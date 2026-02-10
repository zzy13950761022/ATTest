import tensorflow as tf
from tensorflow.python.ops import histogram_ops
import numpy as np

# 测试 value_range[0] > value_range[1] 的情况
print("Testing value_range[0] > value_range[1]...")
try:
    values = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    value_range = tf.constant([10.0, 0.0], dtype=tf.float32)  # 10.0 > 0.0
    result = histogram_ops.histogram_fixed_width_bins(
        values=values,
        value_range=value_range,
        nbins=5,
        dtype=tf.int32
    )
    print(f"Result: {result.numpy()}")
    print("WARNING: No exception raised for value_range[0] > value_range[1]")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting value_range[0] == value_range[1]...")
try:
    values = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    value_range = tf.constant([5.0, 5.0], dtype=tf.float32)  # 5.0 == 5.0
    result = histogram_ops.histogram_fixed_width_bins(
        values=values,
        value_range=value_range,
        nbins=5,
        dtype=tf.int32
    )
    print(f"Result: {result.numpy()}")
    print("WARNING: No exception raised for value_range[0] == value_range[1]")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting valid value_range...")
try:
    values = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    value_range = tf.constant([0.0, 5.0], dtype=tf.float32)  # 0.0 < 5.0
    result = histogram_ops.histogram_fixed_width_bins(
        values=values,
        value_range=value_range,
        nbins=5,
        dtype=tf.int32
    )
    print(f"Result: {result.numpy()}")
    print("SUCCESS: Valid value_range works correctly")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")