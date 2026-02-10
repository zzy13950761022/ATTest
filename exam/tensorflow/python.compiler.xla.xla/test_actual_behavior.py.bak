import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.xla import xla

# 测试单值输出
def single_value_computation(x):
    return tf.reduce_sum(x)

# 创建测试输入
x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

# 直接调用
direct_result = single_value_computation(x)
print(f"Direct result: {direct_result}")
print(f"Direct result type: {type(direct_result)}")

# 使用 xla.compile
compiled_result = xla.compile(single_value_computation, inputs=[x])
print(f"\nCompiled result: {compiled_result}")
print(f"Compiled result type: {type(compiled_result)}")
print(f"Compiled result length: {len(compiled_result) if hasattr(compiled_result, '__len__') else 'N/A'}")

# 测试多值输出
def multi_value_computation(x, y):
    return x + y, x * y

y = tf.constant([2.0, 3.0, 4.0], dtype=tf.float32)
direct_multi = multi_value_computation(x, y)
compiled_multi = xla.compile(multi_value_computation, inputs=[x, y])

print(f"\nMulti direct result: {direct_multi}")
print(f"Multi direct type: {type(direct_multi)}")
print(f"Multi compiled result: {compiled_multi}")
print(f"Multi compiled type: {type(compiled_multi)}")