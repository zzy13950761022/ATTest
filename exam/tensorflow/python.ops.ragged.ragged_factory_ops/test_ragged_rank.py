import tensorflow as tf
from tensorflow.python.ops.ragged import ragged_factory_ops

# 测试 ragged_rank=1 的行为
print("测试 1: ragged_rank=1 的简单案例")
try:
    # 对于 ragged_rank=1，我们需要一个3层嵌套的列表
    # 最外层：不同数量的中层列表
    # 中层：不同数量的内层列表  
    # 内层：相同长度的标量列表
    result = ragged_factory_ops.constant(
        [[[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10], [11, 12, 13]]],
        ragged_rank=1
    )
    print(f"成功: shape={result.shape}, ragged_rank={result.ragged_rank}")
    print(f"flat_values shape: {result.flat_values.shape}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")

print("\n测试 2: 测试计划中的原始数据")
try:
    result = ragged_factory_ops.constant(
        [[[1, 2], [3]], [[4, 5, 6]]],
        ragged_rank=1
    )
    print(f"成功: shape={result.shape}, ragged_rank={result.ragged_rank}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")

print("\n测试 3: 修正后的数据 - 所有内层列表长度一致")
try:
    result = ragged_factory_ops.constant(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        ragged_rank=1
    )
    print(f"成功: shape={result.shape}, ragged_rank={result.ragged_rank}")
    print(f"flat_values shape: {result.flat_values.shape}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")

print("\n测试 4: ragged_rank=2 的案例")
try:
    result = ragged_factory_ops.constant(
        [[[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]],
        ragged_rank=2
    )
    print(f"成功: shape={result.shape}, ragged_rank={result.ragged_rank}")
    print(f"flat_values shape: {result.flat_values.shape}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")