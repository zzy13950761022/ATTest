import tensorflow as tf
from unittest import mock

# 测试不同的mock路径
try:
    with mock.patch('tensorflow.python.ops.gen_parsing_ops') as mock_module:
        print("Mock路径 'tensorflow.python.ops.gen_parsing_ops' 成功")
except Exception as e:
    print(f"Mock路径 'tensorflow.python.ops.gen_parsing_ops' 失败: {e}")

# 测试另一个可能的路径
try:
    with mock.patch('tensorflow.python.ops.parsing_ops.gen_parsing_ops') as mock_module:
        print("Mock路径 'tensorflow.python.ops.parsing_ops.gen_parsing_ops' 成功")
except Exception as e:
    print(f"Mock路径 'tensorflow.python.ops.parsing_ops.gen_parsing_ops' 失败: {e}")

# 检查模块是否存在
try:
    import tensorflow.python.ops.gen_parsing_ops
    print("模块 tensorflow.python.ops.gen_parsing_ops 可以导入")
except ImportError as e:
    print(f"无法导入 tensorflow.python.ops.gen_parsing_ops: {e}")

# 检查parsing_ops中是否有gen_parsing_ops属性
from tensorflow.python.ops import parsing_ops
if hasattr(parsing_ops, 'gen_parsing_ops'):
    print("parsing_ops 有 gen_parsing_ops 属性")
else:
    print("parsing_ops 没有 gen_parsing_ops 属性")