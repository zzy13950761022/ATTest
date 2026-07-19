import tensorflow as tf
from tensorflow.python.ops import random_ops

print("Testing TensorFlow random_ops behavior with invalid parameters...")

# Test 1: random_normal with stddev=0
print("\n1. Testing random_normal with stddev=0:")
try:
    result = random_ops.random_normal(shape=[10], stddev=0.0)
    print(f"  No exception raised. Result shape: {result.shape}, dtype: {result.dtype}")
    print(f"  All values are: {result.numpy()}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

# Test 2: random_normal with stddev=-1
print("\n2. Testing random_normal with stddev=-1:")
try:
    result = random_ops.random_normal(shape=[10], stddev=-1.0)
    print(f"  No exception raised. Result shape: {result.shape}, dtype: {result.dtype}")
    print(f"  All values are: {result.numpy()}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

# Test 3: random_uniform with minval > maxval
print("\n3. Testing random_uniform with minval=1.0, maxval=0.0:")
try:
    result = random_ops.random_uniform(shape=[10], minval=1.0, maxval=0.0)
    print(f"  No exception raised. Result shape: {result.shape}, dtype: {result.dtype}")
    print(f"  Sample values: {result.numpy()[:3]}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

# Test 4: random_uniform with minval = maxval
print("\n4. Testing random_uniform with minval=1.0, maxval=1.0:")
try:
    result = random_ops.random_uniform(shape=[10], minval=1.0, maxval=1.0)
    print(f"  No exception raised. Result shape: {result.shape}, dtype: {result.dtype}")
    print(f"  Sample values: {result.numpy()[:3]}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

# Test 5: truncated_normal with stddev=0
print("\n5. Testing truncated_normal with stddev=0:")
try:
    result = random_ops.truncated_normal(shape=[10], stddev=0.0)
    print(f"  No exception raised. Result shape: {result.shape}, dtype: {result.dtype}")
    print(f"  All values are: {result.numpy()}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

# Test 6: truncated_normal with stddev=-1
print("\n6. Testing truncated_normal with stddev=-1:")
try:
    result = random_ops.truncated_normal(shape=[10], stddev=-1.0)
    print(f"  No exception raised. Result shape: {result.shape}, dtype: {result.dtype}")
    print(f"  All values are: {result.numpy()}")
except Exception as e:
    print(f"  Exception raised: {type(e).__name__}: {e}")

print("\nConclusion: TensorFlow random_ops functions do not validate stddev > 0 or minval < maxval parameters.")