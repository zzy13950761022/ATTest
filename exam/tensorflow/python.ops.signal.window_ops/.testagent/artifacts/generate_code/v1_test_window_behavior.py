import numpy as np
import tensorflow as tf
from tensorflow.python.ops.signal import window_ops

# Test periodic window endpoints
print("Testing periodic window endpoints:")
hann_periodic = window_ops.hann_window(10, periodic=True, dtype=tf.float32)
print(f"Hann periodic (length=10): {hann_periodic.numpy()}")
print(f"Start: {hann_periodic.numpy()[0]}, End: {hann_periodic.numpy()[-1]}")

hamming_periodic = window_ops.hamming_window(10, periodic=True, dtype=tf.float32)
print(f"\nHamming periodic (length=10): {hamming_periodic.numpy()}")
print(f"Start: {hamming_periodic.numpy()[0]}, End: {hamming_periodic.numpy()[-1]}")

# Test symmetric window endpoints
print("\n\nTesting symmetric window endpoints:")
hann_symmetric = window_ops.hann_window(10, periodic=False, dtype=tf.float32)
print(f"Hann symmetric (length=10): {hann_symmetric.numpy()}")
print(f"Start: {hann_symmetric.numpy()[0]}, End: {hann_symmetric.numpy()[-1]}")

# Test kaiser_bessel_derived_window behavior
print("\n\nTesting kaiser_bessel_derived_window:")
for length in [1, 2, 3, 4, 5]:
    try:
        window = window_ops.kaiser_bessel_derived_window(length, beta=12.0, dtype=tf.float32)
        print(f"Length={length}: shape={window.shape}, values={window.numpy()}")
    except Exception as e:
        print(f"Length={length}: Error - {e}")

# Test window_length=0 and negative values
print("\n\nTesting invalid window lengths:")
for length in [0, -1, -5]:
    try:
        window = window_ops.hann_window(length, periodic=True, dtype=tf.float32)
        print(f"Length={length}: shape={window.shape}, values={window.numpy()}")
    except Exception as e:
        print(f"Length={length}: Error type={type(e).__name__}, message={str(e)[:100]}")