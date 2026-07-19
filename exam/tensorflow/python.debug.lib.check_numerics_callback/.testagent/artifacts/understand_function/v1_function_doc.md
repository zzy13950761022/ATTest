# tensorflow.python.debug.lib.check_numerics_callback - 函数说明

## 1. 基本信息
- **FQN**: tensorflow.python.debug.lib.check_numerics_callback
- **模块文件**: `tensorflow/python/debug/lib/check_numerics_callback.py`
- **签名**: enable_check_numerics(stack_height_limit=30, path_length_limit=50)
- **对象类型**: module (包含 enable_check_numerics, disable_check_numerics 函数和 CheckNumericsCallback 类)

## 2. 功能概述
启用统一的数值检查机制，在 TensorFlow 的 eager 执行和 graph 执行中检测 NaN/Infinity 值。当浮点张量包含 NaN 或 Infinity 时，会抛出 `tf.errors.InvalidArgumentError`。该机制是线程局部的。

## 3. 参数说明
- stack_height_limit (int/30): 打印堆栈跟踪的高度限制，仅适用于 tf.function 中的操作
- path_length_limit (int/50): 打印堆栈跟踪中包含的文件路径长度限制，仅适用于 tf.function 中的操作

## 4. 返回值
- 无返回值 (None)
- 副作用：注册操作回调以检查数值

## 5. 文档要点
- 仅对浮点数据类型 (dtype.is_floating) 进行检查
- 忽略特定操作的输出 (IGNORE_OP_OUTPUTS 列表)
- 跳过安全操作以减少开销 (SAFE_OPS 列表)
- 在 TPU 上运行时需要先调用 `tf.config.set_soft_device_placement(True)`
- 幂等性：多次调用与一次调用效果相同
- 线程局部：仅在调用线程中生效

## 6. 源码摘要
- 检查线程局部状态中是否已存在回调实例
- 创建 CheckNumericsCallback 实例（如果不存在）
- 通过 op_callbacks.add_op_callback 注册回调
- 记录日志并增加计数器
- 依赖：op_callbacks, CheckNumericsCallback 类, 监控计数器

## 7. 示例与用法
```python
import tensorflow as tf
tf.debugging.enable_check_numerics()

@tf.function
def square_log_x_plus_1(x):
    v = tf.math.log(x + 1)
    return tf.math.square(v)

x = -1.0
y = square_log_x_plus_1(x)  # 会抛出 InvalidArgumentError
```

## 8. 风险与空白
- 模块包含多个实体：enable_check_numerics, disable_check_numerics, CheckNumericsCallback 类
- 需要测试 enable/disable 的幂等性
- 需要测试线程局部行为
- 需要测试 TPU 环境下的特殊要求
- 需要验证 IGNORE_OP_OUTPUTS 和 SAFE_OPS 列表的覆盖范围
- 缺少对非浮点数据类型的明确处理说明
- 缺少性能开销的具体量化信息