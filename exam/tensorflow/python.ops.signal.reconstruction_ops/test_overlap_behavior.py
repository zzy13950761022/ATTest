import tensorflow as tf
from tensorflow.python.ops.signal.reconstruction_ops import overlap_and_add
import numpy as np

# Test 1: frame_step > frame_length - should this raise an error?
print("Test 1: frame_step > frame_length")
signal = tf.constant(np.random.randn(3, 5).astype(np.float32))
try:
    result = overlap_and_add(signal, 6)  # frame_step=6 > frame_length=5
    print(f"  No error raised. Result shape: {result.shape}")
except Exception as e:
    print(f"  Error raised: {type(e).__name__}: {e}")

# Test 2: rank < 2
print("\nTest 2: rank < 2")
signal = tf.constant(np.random.randn(5).astype(np.float32))
try:
    result = overlap_and_add(signal, 2)
    print(f"  No error raised. Result shape: {result.shape}")
except Exception as e:
    print(f"  Error raised: {type(e).__name__}: {e}")

# Test 3: non-integer frame_step
print("\nTest 3: non-integer frame_step")
signal = tf.constant(np.random.randn(3, 5).astype(np.float32))
try:
    result = overlap_and_add(signal, tf.constant(3.0, dtype=tf.float32))
    print(f"  No error raised. Result shape: {result.shape}")
except Exception as e:
    print(f"  Error raised: {type(e).__name__}: {e}")

# Test 4: non-scalar frame_step
print("\nTest 4: non-scalar frame_step")
signal = tf.constant(np.random.randn(3, 5).astype(np.float32))
try:
    result = overlap_and_add(signal, tf.constant([3, 3], dtype=tf.int32))
    print(f"  No error raised. Result shape: {result.shape}")
except Exception as e:
    print(f"  Error raised: {type(e).__name__}: {e}")