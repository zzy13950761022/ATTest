import numpy as np
import tensorflow as tf
from tensorflow.python.ops.signal import mel_ops

# 使用默认参数
num_mel_bins = 20
num_spectrogram_bins = 129
sample_rate = 8000
lower_edge_hertz = 125.0
upper_edge_hertz = 3800.0
dtype = tf.float32

# 调用目标函数
weight_matrix = mel_ops.linear_to_mel_weight_matrix(
    num_mel_bins=num_mel_bins,
    num_spectrogram_bins=num_spectrogram_bins,
    sample_rate=sample_rate,
    lower_edge_hertz=lower_edge_hertz,
    upper_edge_hertz=upper_edge_hertz,
    dtype=dtype
)

# 转换为numpy数组
weight_np = weight_matrix.numpy()

# 检查每列最大值
column_max = np.max(weight_np, axis=0)
print("每列最大值:", column_max)
print("最小值:", np.min(column_max))
print("最大值:", np.max(column_max))
print("平均值:", np.mean(column_max))

# 检查每列和
column_sum = np.sum(weight_np, axis=0)
print("\n每列和:", column_sum)
print("最小值:", np.min(column_sum))
print("最大值:", np.max(column_sum))
print("平均值:", np.mean(column_sum))

# 检查矩阵特性
print("\n矩阵形状:", weight_np.shape)
print("非零元素比例:", np.count_nonzero(weight_np) / weight_np.size)

# 检查前几列的详细情况
for i in range(min(5, num_mel_bins)):
    col = weight_np[:, i]
    non_zero_indices = np.where(col > 1e-6)[0]
    if len(non_zero_indices) > 0:
        print(f"\n第{i}列:")
        print(f"  最大值: {np.max(col):.6f}")
        print(f"  最大值位置: {np.argmax(col)}")
        print(f"  非零元素数量: {len(non_zero_indices)}")
        print(f"  非零元素范围: {non_zero_indices[0]} - {non_zero_indices[-1]}")
        print(f"  列和: {np.sum(col):.6f}")