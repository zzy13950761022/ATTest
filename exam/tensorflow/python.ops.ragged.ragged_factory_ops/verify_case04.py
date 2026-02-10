import tensorflow as tf
from tensorflow.python.ops.ragged import ragged_factory_ops

print("测试 CASE_04 的第一个参数组合:")
print("pylist = [[[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]]")
print("ragged_rank = 1")
print()

try:
    result = ragged_factory_ops.constant(
        [[[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]],
        ragged_rank=1
    )
    print(f"成功!")
    print(f"shape: {result.shape}")
    print(f"ragged_rank: {result.ragged_rank}")
    print(f"flat_values shape: {result.flat_values.shape}")
    print(f"flat_values: {result.flat_values.numpy()}")
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

print("测试 CASE_04 的第二个参数组合:")
print("pylist = [[[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]]")
print("ragged_rank = 2")
print()

try:
    result = ragged_factory_ops.constant(
        [[[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]],
        ragged_rank=2
    )
    print(f"成功!")
    print(f"shape: {result.shape}")
    print(f"ragged_rank: {result.ragged_rank}")
    print(f"flat_values shape: {result.flat_values.shape}")
    print(f"flat_values: {result.flat_values.numpy()}")
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")