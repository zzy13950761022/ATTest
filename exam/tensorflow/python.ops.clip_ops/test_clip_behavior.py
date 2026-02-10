import numpy as np
import tensorflow as tf
from tensorflow.python.ops import clip_ops

# 测试 clip_min > clip_max 时的实际行为
np.random.seed(42)
input_data = np.random.uniform(0.0, 3.0, size=[2, 2]).astype(np.float32)
t = tf.constant(input_data, dtype=tf.float32)

clip_min = 5.0  # 大于 clip_max
clip_max = 1.0  # 小于 clip_min

print("Input data:")
print(input_data)
print(f"\nclip_min: {clip_min}, clip_max: {clip_max}")

try:
    result = clip_ops.clip_by_value(t, clip_min, clip_max)
    print("\nResult (no exception raised):")
    print(result.numpy())
    print(f"\nAll values are: {np.unique(result.numpy())}")
except Exception as e:
    print(f"\nException raised: {type(e).__name__}: {e}")

# 测试 clip_min == clip_max 的情况
print("\n\nTesting clip_min == clip_max:")
clip_value = 2.5
result2 = clip_ops.clip_by_value(t, clip_value, clip_value)
print(f"Result when clip_min == clip_max == {clip_value}:")
print(result2.numpy())