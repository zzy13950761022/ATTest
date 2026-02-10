import tensorflow as tf
from tensorflow.python.compiler.xla import xla
import numpy as np

# 测试返回None的函数
def none_output_computation(x):
    _ = tf.reduce_sum(x)  # 操作会被执行
    return None

# 测试返回操作和None混合的函数
def mixed_output_computation(x):
    return tf.reduce_sum(x), None

# 创建测试输入
x = tf.constant(np.random.randn(3, 4).astype(np.float32))

print("测试返回None的函数:")
try:
    result = xla.compile(none_output_computation, inputs=[x])
    print(f"成功: {result}")
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")

print("\n测试混合输出的函数:")
try:
    result = xla.compile(mixed_output_computation, inputs=[x])
    print(f"成功: {result}")
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")

print("\n直接调用对比:")
print(f"none_output_computation直接调用: {none_output_computation(x)}")
print(f"mixed_output_computation直接调用: {mixed_output_computation(x)}")