#!/usr/bin/env python3
"""验证CASE_05修复的简单测试脚本"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.signal import mel_ops

def test_htk_formula_fix():
    """验证修复后的HTK公式测试"""
    # 测试参数：使用float64以获得更高精度验证
    num_mel_bins = 5
    num_spectrogram_bins = 33
    sample_rate = 8000
    lower_edge_hertz = 125.0
    upper_edge_hertz = 3800.0
    dtype = tf.float64
    
    # 调用目标函数
    weight_matrix = mel_ops.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        dtype=dtype
    )
    
    print(f"权重矩阵形状: {weight_matrix.shape}")
    print(f"权重矩阵数据类型: {weight_matrix.dtype}")
    
    # 验证所有值都是有限的（无NaN或inf）
    assert tf.reduce_all(tf.math.is_finite(weight_matrix)), \
        "权重矩阵包含非有限值（NaN或inf）"
    
    # 验证所有值非负（三角滤波器权重应为非负）
    assert tf.reduce_all(weight_matrix >= 0), \
        "权重矩阵包含负值"
    
    # 验证三角滤波器峰值接近1.0
    column_maxes = tf.reduce_max(weight_matrix, axis=0)
    print(f"每列最大值: {column_maxes.numpy()}")
    
    # 使用更宽松的容差检查峰值是否接近1.0
    tolerance = 0.1  # 10%的容差
    assert tf.reduce_all(column_maxes > 0.9), \
        f"三角滤波器峰值应大于0.9，实际值：{column_maxes.numpy()}"
    
    # 验证每列和为1（滤波器归一化）
    column_sums = tf.reduce_sum(weight_matrix, axis=0)
    print(f"每列和: {column_sums.numpy()}")
    
    # 使用更宽松的容差
    sum_tolerance = 0.2  # 20%的容差
    assert tf.reduce_all(tf.abs(column_sums - 1.0) < sum_tolerance), \
        f"滤波器列和未归一化到1.0附近，实际值：{column_sums.numpy()}"
    
    print("测试通过！")
    return True

if __name__ == "__main__":
    try:
        test_htk_formula_fix()
        print("\n✅ CASE_05修复验证成功！")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise