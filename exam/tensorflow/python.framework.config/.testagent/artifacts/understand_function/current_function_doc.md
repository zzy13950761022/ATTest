# tensorflow.python.framework.config - 函数说明

## 1. 基本信息
- **FQN**: tensorflow.python.framework.config
- **模块文件**: `D:\Coding\Anaconda\envs\testagent-experiment\lib\site-packages\tensorflow\python\framework\config.py`
- **签名**: 模块（包含多个函数）
- **对象类型**: module

## 2. 功能概述
TensorFlow 执行配置模块，提供硬件、线程、内存、设备等运行时配置功能。包含实验性功能如 TensorFloat-32 精度控制、操作确定性等。

## 3. 参数说明
模块包含多个函数，主要类别：
- 线程配置：`get/set_intra_op_parallelism_threads`, `get/set_inter_op_parallelism_threads`
- 设备管理：`list_physical_devices`, `list_logical_devices`, `get/set_visible_devices`
- 内存管理：`get_memory_info`, `reset_memory_stats`, `get/set_memory_growth`
- 优化器配置：`get/set_optimizer_jit`, `get/set_optimizer_experimental_options`
- 实验功能：`enable/disable_tensor_float_32_execution`, `enable/disable_op_determinism`

## 4. 返回值
各函数返回类型不同：
- 布尔值：状态查询函数
- 整数：线程数配置
- 列表：设备列表
- 字典：内存信息、设备详情
- 无返回值：配置设置函数

## 5. 文档要点
- TensorFloat-32 仅支持 NVIDIA Ampere 及以上 GPU
- 内存统计仅支持 GPU 和 TPU，不支持 CPU
- 设备配置需在运行时初始化前完成
- 操作确定性会降低性能，主要用于调试

## 6. 源码摘要
- 关键依赖：`context.context()` 提供底层配置接口
- 外部包装：`_pywrap_tensor_float_32_execution`, `_pywrap_determinism`
- 副作用：修改全局运行时状态（线程数、设备可见性、内存策略）
- 分支逻辑：设备类型检查、运行时状态验证

## 7. 示例与用法
```python
# TensorFloat-32 控制
tf.config.experimental.enable_tensor_float_32_execution(False)

# 设备列表查询
gpus = tf.config.list_physical_devices('GPU')

# 线程配置
tf.config.threading.set_intra_op_parallelism_threads(4)

# 内存信息
mem_info = tf.config.experimental.get_memory_info('GPU:0')
```

## 8. 风险与空白
- 多实体模块：包含 30+ 函数，需分别测试
- 硬件依赖：部分功能需要特定硬件（GPU/TPU）
- 运行时限制：某些配置必须在初始化前设置
- 实验性功能：API 可能变更，缺少完整类型注解
- 缺少信息：部分函数文档未明确异常类型
- 测试挑战：需要模拟不同硬件环境
- 边界情况：内存限制、设备不存在、无效参数处理