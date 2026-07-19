# tensorflow.python.data.experimental.ops.batching 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证数据集批处理转换操作，包括密集张量到RaggedTensor/SparseTensor的转换，支持不同形状张量的批处理
- 不在范围内的内容：已弃用的`map_and_batch`、`map_and_batch_with_legacy_function`、`unbatch`函数，以及底层内部类实现细节

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - `batch_size`：tf.int64标量，必须为正整数，无默认值
  - `drop_remainder`：tf.bool标量，默认False
  - `row_splits_dtype`：dtype，默认int64
  - `row_shape`：TensorShape/int64向量，用于稀疏批处理
  - `map_func`：函数，用于map_and_batch操作

- 有效取值范围/维度/设备要求：
  - batch_size必须为正整数
  - row_splits_dtype必须是有效的整数dtype
  - 输入数据集元素必须具有相同rank作为row_shape
  - 每个维度大小必须小于或等于row_shape

- 必需与可选组合：
  - batch_size为必需参数
  - drop_remainder、row_splits_dtype为可选参数
  - row_shape为dense_to_sparse_batch必需参数

- 随机性/全局状态要求：无随机性，无全局状态修改

## 3. 输出与判定
- 期望返回结构及关键字段：返回Dataset转换函数，接受Dataset参数，返回转换后的Dataset
- 容差/误差界（如浮点）：无浮点容差要求，关注形状和类型匹配
- 状态变化或副作用检查点：无I/O或全局状态副作用，纯数据转换

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - batch_size非正整数触发异常
  - 无效row_splits_dtype触发异常
  - 输入元素rank与row_shape不匹配触发异常
  - 维度大小超过row_shape触发异常

- 边界值（空、None、0长度、极端形状/数值）：
  - batch_size=1边界情况
  - 空数据集处理
  - 极端形状张量（如超大维度）
  - 零长度张量批处理

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：仅依赖TensorFlow运行时，无外部资源
- 需要mock/monkeypatch的部分：无需mock，纯函数测试

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. dense_to_ragged_batch基本功能验证
  2. dense_to_sparse_batch形状约束检查
  3. drop_remainder参数行为验证
  4. 不同形状张量的批处理转换
  5. row_splits_dtype参数有效性测试

- 可选路径（中/低优先级合并为一组列表）：
  - 已弃用函数兼容性测试
  - 极端批大小（如接近内存限制）
  - 混合数据类型批处理
  - 嵌套结构批处理
  - 性能基准测试

- 已知风险/缺失信息（仅列条目，不展开）：
  - 内部类`_DenseToRaggedDataset`文档不完整
  - 已弃用函数仍在使用
  - 缺少row_splits_dtype参数边界测试
  - 需要验证不同设备（CPU/GPU）行为一致性