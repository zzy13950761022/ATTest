import tensorflow as tf
from tensorflow.python.ops import histogram_ops

print("Testing histogram_fixed_width_bins with integer dtype...")

# 测试 int32 类型
print("\n1. Testing int32 dtype:")
try:
    values = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    value_range = tf.constant([0, 10], dtype=tf.int32)
    print(f"values dtype: {values.dtype}, value_range dtype: {value_range.dtype}")
    result = histogram_ops.histogram_fixed_width_bins(values, value_range, nbins=5)
    print(f"Success! Result: {result.numpy()}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

# 测试 float32 类型
print("\n2. Testing float32 dtype:")
try:
    values = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)
    value_range = tf.constant([0.0, 10.0], dtype=tf.float32)
    print(f"values dtype: {values.dtype}, value_range dtype: {value_range.dtype}")
    result = histogram_ops.histogram_fixed_width_bins(values, value_range, nbins=5)
    print(f"Success! Result: {result.numpy()}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

# 测试将 int32 转换为 float32
print("\n3. Testing int32 values with float32 value_range:")
try:
    values = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    value_range = tf.constant([0.0, 10.0], dtype=tf.float32)
    print(f"values dtype: {values.dtype}, value_range dtype: {value_range.dtype}")
    result = histogram_ops.histogram_fixed_width_bins(values, value_range, nbins=5)
    print(f"Success! Result: {result.numpy()}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

# 测试 histogram_fixed_width 函数
print("\n4. Testing histogram_fixed_width with int32 dtype:")
try:
    values = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    value_range = tf.constant([0, 10], dtype=tf.int32)
    print(f"values dtype: {values.dtype}, value_range dtype: {value_range.dtype}")
    result = histogram_ops.histogram_fixed_width(values, value_range, nbins=5)
    print(f"Success! Result: {result.numpy()}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")