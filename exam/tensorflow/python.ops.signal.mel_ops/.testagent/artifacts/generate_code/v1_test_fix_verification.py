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
print("每列最大值范围:", np.min(column_max), "到", np.max(column_max))
print("所有列最大值 > 0.8?", np.all(column_max > 0.8))

# 检查修复后的断言
FLOAT32_TOL = 1e-6

# 1. 检查形状
assert weight_matrix.shape == (num_spectrogram_bins, num_mel_bins)

# 2. 检查数据类型
assert weight_matrix.dtype == dtype

# 3. 检查有限性
assert tf.reduce_all(tf.math.is_finite(weight_matrix))

# 4. 检查非负性
assert tf.reduce_all(weight_matrix >= 0.0)

# 5. 检查每列最大值 > 0.8
assert np.all(column_max > 0.8)

# 6. 检查每列和 > 0
column_sum = np.sum(weight_np, axis=0)
assert np.all(column_sum > 0)

# 7. 检查稀疏性
non_zero_ratio = np.count_nonzero(weight_np) / weight_np.size
assert 0.1 < non_zero_ratio < 0.5

# 8. 检查三角滤波器特性
for i in range(1, num_mel_bins - 1):
    col = weight_np[:, i]
    non_zero_indices = np.where(col > FLOAT32_TOL)[0]
    if len(non_zero_indices) > 0:
        # 非零元素应连续
        assert np.all(np.diff(non_zero_indices) == 1)
        
        # 找到最大值位置
        max_idx = np.argmax(col)
        # 检查最大值位置在非零范围内
        assert max_idx in non_zero_indices
        
        # 检查左右两侧是否递减（三角形状）
        if max_idx > non_zero_indices[0]:
            left_diff = np.diff(col[non_zero_indices[0]:max_idx+1])
            assert np.all(left_diff >= -FLOAT32_TOL)
        
        if max_idx < non_zero_indices[-1]:
            right_diff = np.diff(col[max_idx:non_zero_indices[-1]+1])
            assert np.all(right_diff <= FLOAT32_TOL)

print("所有断言通过！")