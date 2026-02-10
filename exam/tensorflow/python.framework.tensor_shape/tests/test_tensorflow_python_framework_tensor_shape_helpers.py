"""
测试 tensorflow.python.framework.tensor_shape 模块中的辅助函数
"""
import pytest
import tensorflow as tf
from tensorflow.python.framework import tensor_shape

# ==== BLOCK:HEADER START ====
# 测试辅助函数和 V1/V2 兼容性切换
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_05 START ====
@pytest.mark.parametrize(
    "func,input_val,expected,expected_type",
    [
        ("dimension_value", 5, 5, None),
        ("as_dimension", 3, None, "Dimension"),
        ("as_shape", [2, 3], None, "TensorShape"),
    ]
)
def test_helper_functions_basic(func, input_val, expected, expected_type):
    """测试辅助函数基本功能"""
    
    if func == "dimension_value":
        # 测试 dimension_value 函数
        result = tensor_shape.dimension_value(input_val)
        assert result == expected
        
        # 也测试传入 Dimension 对象
        dim = tensor_shape.Dimension(input_val)
        result_from_dim = tensor_shape.dimension_value(dim)
        assert result_from_dim == expected
        
        # 测试传入 None
        result_none = tensor_shape.dimension_value(None)
        assert result_none is None
        
    elif func == "as_dimension":
        # 测试 as_dimension 函数
        result = tensor_shape.as_dimension(input_val)
        
        # 验证返回类型
        assert isinstance(result, tensor_shape.Dimension)
        
        # 验证值
        assert result.value == input_val
        
        # 测试传入 Dimension 对象（应该返回相同的对象）
        dim = tensor_shape.Dimension(input_val)
        result_from_dim = tensor_shape.as_dimension(dim)
        assert result_from_dim is dim  # 应该返回相同的对象
        
        # 测试传入 None
        result_none = tensor_shape.as_dimension(None)
        assert isinstance(result_none, tensor_shape.Dimension)
        assert result_none.value is None
        
    elif func == "as_shape":
        # 测试 as_shape 函数
        result = tensor_shape.as_shape(input_val)
        
        # 验证返回类型
        assert isinstance(result, tensor_shape.TensorShape)
        
        # 验证形状
        assert result.rank == len(input_val)
        for i, dim in enumerate(result.dims):
            assert dim.value == input_val[i]
        
        # 测试传入 TensorShape 对象（应该返回相同的对象）
        ts = tensor_shape.TensorShape(input_val)
        result_from_ts = tensor_shape.as_shape(ts)
        assert result_from_ts is ts  # 应该返回相同的对象
        
        # 测试传入 None
        result_none = tensor_shape.as_shape(None)
        assert isinstance(result_none, tensor_shape.TensorShape)
        assert result_none.rank is None
        assert result_none.dims is None
        
    else:
        pytest.fail(f"未知函数: {func}")
    
    # 验证函数的幂等性（对于某些函数）
    if func in ["as_dimension", "as_shape"]:
        # 应用两次应该得到相同的结果
        result1 = getattr(tensor_shape, func)(input_val)
        result2 = getattr(tensor_shape, func)(result1)
        
        if func == "as_dimension":
            # 对于 as_dimension，第二次调用应该返回相同的对象
            assert result2 is result1
        elif func == "as_shape":
            # 对于 as_shape，第二次调用应该返回相同的对象
            assert result2 is result1
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_10 START ====
# 辅助函数高级功能测试（deferred）
# ==== BLOCK:CASE_10 END ====

# ==== BLOCK:FOOTER START ====
# 测试结束
# ==== BLOCK:FOOTER END ====