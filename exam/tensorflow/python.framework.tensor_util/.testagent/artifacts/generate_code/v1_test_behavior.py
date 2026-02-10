import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes

# Test 1: Scalar with shape [2,2] and allow_broadcast=False
print("Test 1: Scalar with shape [2,2], allow_broadcast=False")
try:
    proto = tensor_util.make_tensor_proto(
        values=5,
        dtype=dtypes.int32,
        shape=[2, 2],
        verify_shape=False,
        allow_broadcast=False
    )
    print("  Success! No exception raised.")
    print(f"  Shape: {[dim.size for dim in proto.tensor_shape.dim]}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")

# Test 2: List [7] with shape [2,2] and allow_broadcast=False
print("\nTest 2: List [7] with shape [2,2], allow_broadcast=False")
try:
    proto = tensor_util.make_tensor_proto(
        values=[7],
        dtype=dtypes.int32,
        shape=[2, 2],
        verify_shape=False,
        allow_broadcast=False
    )
    print("  Success! No exception raised.")
    print(f"  Shape: {[dim.size for dim in proto.tensor_shape.dim]}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")

# Test 3: List [7] with shape [2,2] and allow_broadcast=True
print("\nTest 3: List [7] with shape [2,2], allow_broadcast=True")
try:
    proto = tensor_util.make_tensor_proto(
        values=[7],
        dtype=dtypes.int32,
        shape=[2, 2],
        verify_shape=False,
        allow_broadcast=True
    )
    print("  Success! No exception raised.")
    print(f"  Shape: {[dim.size for dim in proto.tensor_shape.dim]}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")

# Test 4: Scalar with shape [2,2] and allow_broadcast=True
print("\nTest 4: Scalar with shape [2,2], allow_broadcast=True")
try:
    proto = tensor_util.make_tensor_proto(
        values=5,
        dtype=dtypes.int32,
        shape=[2, 2],
        verify_shape=False,
        allow_broadcast=True
    )
    print("  Success! No exception raised.")
    print(f"  Shape: {[dim.size for dim in proto.tensor_shape.dim]}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")