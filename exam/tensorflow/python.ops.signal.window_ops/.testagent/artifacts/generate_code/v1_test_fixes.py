import numpy as np
import tensorflow as tf
from tensorflow.python.ops.signal import window_ops

# Test CASE_01 fix: periodic window endpoints
print("=== Testing CASE_01 fix ===")
hann_periodic = window_ops.hann_window(10, periodic=True, dtype=tf.float32)
print(f"Hann periodic start: {hann_periodic.numpy()[0]:.6f}, end: {hann_periodic.numpy()[-1]:.6f}")
print(f"Start > 0 and < 0.2: {0 < hann_periodic.numpy()[0] < 0.2}")
print(f"End > 0 and < 0.2: {0 < hann_periodic.numpy()[-1] < 0.2}")

hamming_periodic = window_ops.hamming_window(10, periodic=True, dtype=tf.float32)
print(f"\nHamming periodic start: {hamming_periodic.numpy()[0]:.6f}, end: {hamming_periodic.numpy()[-1]:.6f}")
print(f"Start > 0 and < 0.2: {0 < hamming_periodic.numpy()[0] < 0.2}")
print(f"End > 0 and < 0.2: {0 < hamming_periodic.numpy()[-1] < 0.2}")

# Test CASE_02 fix: kaiser_bessel_derived_window
print("\n\n=== Testing CASE_02 fix ===")
for length in [1, 2, 3, 4, 5, 6]:
    window = window_ops.kaiser_bessel_derived_window(length, beta=12.0, dtype=tf.float32)
    expected_length = (length // 2) * 2
    print(f"Length={length}: shape={window.shape}, expected={expected_length}, match={window.shape[0] == expected_length}")

# Test CASE_03 fix: parameter validation
print("\n\n=== Testing CASE_03 fix ===")
print("Testing invalid dtypes:")
try:
    window_ops.hann_window(10, periodic=True, dtype=tf.int32)
    print("  No error raised for int32 dtype")
except ValueError as e:
    print(f"  ValueError raised: {str(e)[:50]}...")

print("\nTesting window_length=0:")
try:
    window_ops.hamming_window(0, periodic=True, dtype=tf.float32)
    print("  No error raised for window_length=0")
except tf.errors.InvalidArgumentError as e:
    print(f"  InvalidArgumentError raised: {str(e)[:50]}...")
except Exception as e:
    print(f"  Other error raised: {type(e).__name__}: {str(e)[:50]}...")

print("\nTesting window_length=-1:")
try:
    window_ops.hann_window(-1, periodic=True, dtype=tf.float32)
    print("  No error raised for window_length=-1")
except tf.errors.InvalidArgumentError as e:
    print(f"  InvalidArgumentError raised: {str(e)[:50]}...")
except Exception as e:
    print(f"  Other error raised: {type(e).__name__}: {str(e)[:50]}...")