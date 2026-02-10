# tensorflow.python.data.experimental.ops.parsing_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证 `parse_example_dataset` 函数将序列化 Example protos 数据集转换为张量字典数据集
  - 支持 FixedLenFeature、VarLenFeature、SparseFeature、RaggedFeature 特征类型
  - 返回数据集转换函数，可通过 `dataset.apply()` 调用
  - 支持并行解析和确定性控制
- 不在范围内的内容
  - 不验证 protobuf 格式正确性
  - 不测试底层 C++ 操作 `gen_experimental_dataset_ops.parse_example_dataset_v2`
  - 不覆盖 SparseFeature 和带分区 RaggedFeature 的额外映射步骤

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - features: dict/必需，特征键到特征对象的映射字典
  - num_parallel_calls: tf.int32/默认1，并行解析进程数
  - deterministic: bool/默认None，是否保持确定性
- 有效取值范围/维度/设备要求
  - features 不能为 None 或空字典
  - 输入数据集必须是字符串向量数据集（element_spec 为 [None] 的字符串张量）
  - num_parallel_calls 必须为正整数
- 必需与可选组合
  - features 为必需参数
  - num_parallel_calls 和 deterministic 为可选参数
- 随机性/全局状态要求
  - deterministic=None 时使用数据集选项的默认值（True）
  - deterministic=True 时保持顺序
  - deterministic=False 时允许乱序以提高性能

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回类型：数据集转换函数
  - 函数接受 Dataset 参数，返回转换后的 Dataset
  - 输出数据集元素为特征键到张量的映射字典
- 容差/误差界（如浮点）
  - 浮点数值比较容差：1e-6
  - 字符串比较需完全匹配
- 状态变化或副作用检查点
  - 验证输入数据集未被修改
  - 验证返回函数为纯函数（无副作用）

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - features=None 时抛出 ValueError
  - features 为空字典时抛出 ValueError
  - 输入数据集非字符串向量时抛出 InvalidArgumentError
  - num_parallel_calls 非正整数时抛出 ValueError
- 边界值（空、None、0 长度、极端形状/数值）
  - 空字符串向量数据集
  - 包含无效 protobuf 的字符串向量
  - 极端形状：超大字符串、极小字符串
  - 极端数值：浮点溢出、NaN、Inf

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - 依赖 TensorFlow C++ 扩展库
  - 需要 protobuf 解析能力
- 需要 mock/monkeypatch 的部分
  - `gen_experimental_dataset_ops.parse_example_dataset_v2` 操作
  - `parsing_ops._ParseOpParams.from_features` 内部方法
  - `parsing_ops._prepend_none_dimension` 内部函数

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 基本功能：FixedLenFeature 类型解析
  2. 参数验证：features=None 时抛出 ValueError
  3. 并行解析：num_parallel_calls>1 时的正确性
  4. 确定性控制：deterministic 参数三种状态
  5. 多种特征类型：VarLenFeature、SparseFeature、RaggedFeature
- 可选路径（中/低优先级合并为一组列表）
  - 空字符串向量数据集处理
  - 无效 protobuf 字符串的容错性
  - 极端形状和数值的边界测试
  - 内存使用和性能基准
  - 与其他数据集操作的组合使用
- 已知风险/缺失信息（仅列条目，不展开）
  - deterministic=None 时具体行为依赖数据集选项
  - SparseFeature 和带分区 RaggedFeature 需要额外映射步骤
  - 内部依赖 `parsing_ops._ParseOpParams` 等未公开 API
  - 并行解析可能引入非确定性行为