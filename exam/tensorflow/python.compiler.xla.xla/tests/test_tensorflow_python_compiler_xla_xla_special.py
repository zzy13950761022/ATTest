# ==== BLOCK:HEADER START ====
"""
测试 tensorflow.python.compiler.xla.xla 模块的特殊场景与边界处理
"""
import math
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.xla import xla

# 设置随机种子以确保可重复性
np.random.seed(42)
tf.random.set_seed(42)

# 浮点比较容差
RTOL = 1e-6
ATOL = 1e-8

# 辅助函数
def create_test_tensor(shape, dtype=np.float32):
    """创建测试张量"""
    if dtype == np.float32 or dtype == np.float64:
        data = np.random.randn(*shape).astype(dtype)
    elif dtype == np.int32:
        data = np.random.randint(-10, 10, size=shape, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return tf.convert_to_tensor(data, dtype=dtype)

def assert_tensors_equal(t1, t2, rtol=RTOL, atol=ATOL):
    """断言两个张量相等（考虑容差）"""
    if isinstance(t1, tf.Tensor) and isinstance(t2, tf.Tensor):
        np.testing.assert_allclose(t1.numpy(), t2.numpy(), rtol=rtol, atol=atol)
    elif isinstance(t1, (list, tuple)) and isinstance(t2, (list, tuple)):
        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert_tensors_equal(a, b, rtol, atol)
    else:
        assert t1 == t2
# ==== BLOCK:HEADER END ====

class TestXLACompileSpecial:
    """测试 xla.compile 函数的特殊场景与边界处理"""
    
    # ==== BLOCK:CASE_03 START ====
    @pytest.mark.parametrize("dtype,shape", [
        (np.float32, (3,)),  # 基础测试用例
        (np.int32, (4,)),  # 整数类型的单值输出
    ])
    def test_single_value_output_wrapping(self, dtype, shape):
        """测试单值输出包装测试"""
        
        # 定义返回单值的函数
        def single_value_computation(x):
            # 返回单个张量（不是元组）
            return tf.reduce_sum(x)
        
        # 创建测试输入
        x = create_test_tensor(shape, dtype)
        
        # 直接调用计算函数作为基准
        direct_result = single_value_computation(x)
        
        # 使用 xla.compile 编译计算
        compiled_result = xla.compile(single_value_computation, inputs=[x])
        
        # 验证输出类型
        # 根据实际行为，xla.compile 返回列表而不是元组
        assert isinstance(compiled_result, list), "xla.compile 应该返回列表"
        assert len(compiled_result) == 1, "列表应该只包含一个元素"
        
        # 获取包装后的值
        wrapped_value = compiled_result[0]
        
        # 验证包装值的类型
        assert isinstance(wrapped_value, tf.Tensor), "包装值应该是张量"
        
        # 验证输出形状
        # reduce_sum 返回标量，所以形状应该是 ()
        assert wrapped_value.shape == direct_result.shape, "输出形状应该一致"
        assert wrapped_value.shape == (), "单值输出应该是标量"
        
        # 验证输出数据类型
        assert wrapped_value.dtype == direct_result.dtype, "输出数据类型应该一致"
        
        # 验证值相等性
        np.testing.assert_allclose(
            wrapped_value.numpy(),
            direct_result.numpy(),
            rtol=RTOL,
            atol=ATOL,
            err_msg="单值输出数值不匹配"
        )
        
        # 验证列表包装的正确性
        # 确保编译结果确实是列表，而不是单个张量
        assert not isinstance(compiled_result, tf.Tensor), "编译结果不应该是单个张量"
        
        # 验证可以直接解包使用
        unpacked_value = compiled_result[0]
        assert_tensors_equal(unpacked_value, direct_result)
        
        # 验证文档中提到的特殊处理：单值输出应该被包装
        # 直接调用返回单个张量，但 xla.compile 返回包含该张量的列表
        assert isinstance(direct_result, tf.Tensor), "直接调用应该返回张量"
        assert isinstance(compiled_result, list), "xla.compile 应该返回列表"
    # ==== BLOCK:CASE_03 END ====
    
    # ==== BLOCK:CASE_07 START ====
    def test_none_output_handling(self):
        """测试None输出处理"""
        
        # 定义返回None的函数
        def none_output_computation(x):
            # 执行操作但不返回张量
            _ = tf.reduce_sum(x)  # 操作会被执行
            return None
        
        # 创建测试输入
        x = create_test_tensor((3, 4), np.float32)
        
        # 直接调用计算函数
        # 注意：直接调用返回None
        direct_result = none_output_computation(x)
        assert direct_result is None, "直接调用应该返回None"
        
        # 使用 xla.compile 编译计算 - 预期会失败
        # 根据文档，None输出应该返回一个NoOp操作
        # 但实际上由于tf.function的限制会抛出ValueError
        with pytest.raises(ValueError) as exc_info:
            xla.compile(none_output_computation, inputs=[x])
        
        # 验证错误消息
        error_msg = str(exc_info.value)
        assert "None values not supported" in error_msg or \
               "must all either be Operations or convertible to Tensors" in error_msg, \
               f"错误消息不符合预期: {error_msg}"
        
        # 验证操作可以执行（不会抛出异常）
        # 在TensorFlow中，操作需要在会话中执行，但在eager模式下会自动执行
        # 这里我们只验证操作存在且类型正确
        
        # 测试带有操作和None混合输出的函数
        def mixed_output_computation(x):
            # 返回一个张量和一个None
            return tf.reduce_sum(x), None
        
        # 直接调用
        direct_mixed = mixed_output_computation(x)
        assert isinstance(direct_mixed, tuple), "直接调用应该返回元组"
        assert len(direct_mixed) == 2, "应该有两个输出"
        assert isinstance(direct_mixed[0], tf.Tensor), "第一个输出应该是张量"
        assert direct_mixed[1] is None, "第二个输出应该是None"
        
        # 使用 xla.compile - 预期会失败，因为包含None输出
        with pytest.raises(ValueError) as exc_info:
            xla.compile(mixed_output_computation, inputs=[x])
        
        # 验证错误消息
        error_msg = str(exc_info.value)
        assert "None values not supported" in error_msg or \
               "must all either be Operations or convertible to Tensors" in error_msg, \
               f"错误消息不符合预期: {error_msg}"
        
        # 注意：由于tf.function的限制，包含None输出的函数无法被编译
        # 文档中提到的行为可能只适用于非eager模式或旧版本
        
        # 测试其他边界情况：返回空列表（零个输出）
        def empty_output_computation(x):
            # 执行操作但不返回任何值
            _ = tf.reduce_sum(x)
            return []
        
        # 直接调用
        direct_empty = empty_output_computation(x)
        assert isinstance(direct_empty, list), "直接调用应该返回列表"
        assert len(direct_empty) == 0, "应该返回空列表"
        
        # 使用 xla.compile 编译
        compiled_empty = xla.compile(empty_output_computation, inputs=[x])
        
        # 验证编译结果
        assert isinstance(compiled_empty, list), "应该返回列表"
        assert len(compiled_empty) == 0, "应该返回空列表"
        
        # 测试返回NoOp操作的情况（文档中的"Operation-only outputs"）
        def noop_computation(x):
            with tf.control_dependencies([x]):
                return tf.no_op()
        
        # 直接调用
        direct_noop = noop_computation(x)
        assert isinstance(direct_noop, tf.Operation), "直接调用应该返回Operation"
        assert direct_noop.type == "NoOp", "应该返回NoOp操作"
        
        # 使用 xla.compile 编译
        compiled_noop = xla.compile(noop_computation, inputs=[x])
        
        # 验证编译结果
        assert isinstance(compiled_noop, list), "应该返回列表"
        assert len(compiled_noop) == 1, "应该有一个输出"
        assert isinstance(compiled_noop[0], tf.Operation), "输出应该是Operation"
        assert compiled_noop[0].type == "NoOp", "应该返回NoOp操作"
    # ==== BLOCK:CASE_07 END ====
    
    # ==== BLOCK:CASE_08 START ====
    # ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====