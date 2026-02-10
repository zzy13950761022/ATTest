"""
测试 tensorflow.python.framework.tensor_shape 模块中的 TensorShape 类
"""
import pytest
import tensorflow as tf
from tensorflow.python.framework import tensor_shape

# ==== BLOCK:HEADER START ====
# 测试 TensorShape 类的基本功能
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize(
    "dims,expected_rank,expected_dims",
    [
        ([2, 3], 2, [2, 3]),
        ([], 0, []),
        (None, None, None),
    ]
)
def test_tensorshape_construction(dims, expected_rank, expected_dims):
    """测试 TensorShape 基本构造"""
    # 创建 TensorShape 对象
    shape = tensor_shape.TensorShape(dims)
    
    # 验证 rank 属性
    assert shape.rank == expected_rank
    
    # 验证 dims 属性
    if expected_dims is None:
        assert shape.dims is None
    else:
        assert shape.dims is not None
        assert len(shape.dims) == len(expected_dims)
        
        # 验证每个维度的值
        for i, (actual_dim, expected_value) in enumerate(zip(shape.dims, expected_dims)):
            if expected_value is None:
                assert actual_dim.value is None
            else:
                assert actual_dim.value == expected_value
    
    # 验证 is_fully_defined 属性
    if dims is None:
        assert not shape.is_fully_defined
    elif any(d is None for d in dims):
        assert not shape.is_fully_defined
    else:
        assert shape.is_fully_defined
    
    # 验证与列表/元组的比较
    if dims is not None:
        # TensorShape 应该与原始列表相等
        assert shape == dims
        assert dims == shape
        
        # 也应该与元组相等
        assert shape == tuple(dims)
        assert tuple(dims) == shape
    
    # 验证空形状
    if dims == []:
        assert shape.rank == 0
        assert len(shape) == 0
        assert shape.is_fully_defined
    
    # 验证完全未知形状
    if dims is None:
        assert shape.rank is None
        assert shape.dims is None
        assert not shape.is_fully_defined
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
@pytest.mark.parametrize(
    "shape1,shape2,expected",
    [
        ([2, 3], [2, 3], True),
        ([2, None], [2, 5], True),
        ([2, 3], [3, 2], False),
    ]
)
def test_shape_compatibility_check(shape1, shape2, expected):
    """测试形状兼容性检查"""
    # 创建 TensorShape 对象
    ts1 = tensor_shape.TensorShape(shape1)
    ts2 = tensor_shape.TensorShape(shape2)
    
    # 测试兼容性
    result = ts1.is_compatible_with(ts2)
    assert result == expected
    
    # 验证对称性：is_compatible_with 应该是对称的
    # 注意：兼容性检查通常是对称的，但需要验证
    symmetric_result = ts2.is_compatible_with(ts1)
    assert symmetric_result == expected, "兼容性检查应该是对称的"
    
    # 验证与自身的兼容性
    assert ts1.is_compatible_with(ts1), "形状应该与自身兼容"
    assert ts2.is_compatible_with(ts2), "形状应该与自身兼容"
    
    # 验证与原始列表的兼容性
    list_result = ts1.is_compatible_with(shape2)
    assert list_result == expected, "应该能够与列表进行兼容性检查"
    
    # 验证与元组的兼容性
    tuple_result = ts1.is_compatible_with(tuple(shape2))
    assert tuple_result == expected, "应该能够与元组进行兼容性检查"
    
    # 对于已知维度，验证具体值
    if expected and shape1 is not None and shape2 is not None:
        # 如果形状兼容，检查每个维度
        for i, (d1, d2) in enumerate(zip(shape1, shape2)):
            if d1 is not None and d2 is not None:
                # 如果两个维度都已知，它们必须相等
                assert d1 == d2, f"维度 {i}: {d1} != {d2}"
            elif d1 is None or d2 is None:
                # 如果至少一个维度未知，兼容性应该为 True
                pass
            else:
                # 两个维度都未知，兼容性应该为 True
                pass
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_08 START ====
# TensorShape 高级功能测试（deferred）
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:CASE_09 START ====
# TensorShape 边缘情况测试（deferred）
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:FOOTER START ====
# 测试结束
# ==== BLOCK:FOOTER END ====