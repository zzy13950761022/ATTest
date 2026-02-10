import tensorflow as tf
import numpy as np

# Test frame function with frame_length=0
signal = tf.constant(np.random.randn(100).astype(np.float32))

try:
    result = tf.signal.frame(signal, frame_length=0, frame_step=16)
    print(f"frame_length=0 succeeded, result shape: {result.shape}")
except Exception as e:
    print(f"frame_length=0 failed with error: {type(e).__name__}: {e}")

try:
    result = tf.signal.frame(signal, frame_length=32, frame_step=0)
    print(f"frame_step=0 succeeded, result shape: {result.shape}")
except Exception as e:
    print(f"frame_step=0 failed with error: {type(e).__name__}: {e}")

try:
    result = tf.signal.frame(signal, frame_length=-1, frame_step=16)
    print(f"frame_length=-1 succeeded, result shape: {result.shape}")
except Exception as e:
    print(f"frame_length=-1 failed with error: {type(e).__name__}: {e}")