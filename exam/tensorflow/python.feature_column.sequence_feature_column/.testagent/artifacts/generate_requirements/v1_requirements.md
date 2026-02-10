# tensorflow.python.feature_column.sequence_feature_column 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证6个序列特征列函数正确创建序列特征列对象
  - 确保序列数据形状转换符合RNN输入要求
  - 验证特征列与embedding_column/indicator_column的兼容性
  - 检查输入验证逻辑（rank、dtype、参数范围）
- 不在范围内的内容
  - embedding_column/indicator_column的具体实现
  - SequenceFeatures层的内部逻辑
  - 模型训练和推理性能

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - key: str，特征名称标识符
  - num_buckets: int ≥ 1，分类桶数量
  - hash_bucket_size: int ≥ 1，哈希桶大小
  - vocabulary_file: str，词汇表文件路径
  - vocabulary_size: int，词汇表大小
  - vocabulary_list: iterable，词汇列表
  - num_oov_buckets: int ≥ 0，OOV桶数量
  - default_value: int/float，默认值
  - shape: tuple，数值列形状
  - dtype: dtypes，数据类型（int64/string/float32）
  - normalizer_fn: callable，归一化函数
- 有效取值范围/维度/设备要求
  - num_buckets ≥ 1，hash_bucket_size ≥ 1
  - vocabulary_size ≥ 1，num_oov_buckets ≥ 0
  - shape元组元素必须为正整数
  - dtype限制：int64、string、float32
- 必需与可选组合
  - key为必需参数
  - default_value与num_oov_buckets互斥
  - vocabulary_file需要vocabulary_size
- 随机性/全局状态要求
  - 无随机性要求
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - concatenate_context_input: float32 Tensor，形状[batch_size, padded_length, d0+d1]
  - 其他函数：SequenceCategoricalColumn或SequenceNumericColumn实例
  - 实例必须包含key、dtype、num_buckets等属性
- 容差/误差界（如浮点）
  - 浮点计算使用TensorFlow默认精度
  - 形状转换必须精确匹配
- 状态变化或副作用检查点
  - 无文件I/O副作用
  - 无全局状态修改
  - 纯函数行为

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - key非字符串类型
  - num_buckets < 1
  - hash_bucket_size ≤ 1
  - vocabulary_size < 1
  - num_oov_buckets < 0
  - shape包含非正整数
  - dtype不支持的类型
  - default_value与num_oov_buckets同时指定
- 边界值（空、None、0长度、极端形状/数值）
  - key为空字符串
  - vocabulary_list为空迭代器
  - shape为()或包含0的元组
  - 极大num_buckets/hash_bucket_size值
  - 负default_value

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - vocabulary_file需要真实文件路径
  - 依赖TensorFlow运行时环境
  - 需要GPU/CPU设备支持张量运算
- 需要mock/monkeypatch的部分
  - vocabulary_file文件读取操作
  - TensorFlow内部feature_column_v2模块
  - 设备分配逻辑

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. 所有6个函数基本参数组合创建特征列
  2. concatenate_context_input形状拼接正确性
  3. default_value与num_oov_buckets互斥验证
  4. 参数边界值异常触发
  5. 特征列与embedding_column集成测试
- 可选路径（中/低优先级合并为一组列表）
  - 多批次不同形状输入处理
  - 极端词汇表大小性能测试
  - 不同dtype组合兼容性
  - normalizer_fn自定义函数集成
  - 序列长度变化场景
  - 批量大小边界测试
- 已知风险/缺失信息（仅列条目，不展开）
  - API处于开发中，可能频繁变更
  - 部分参数类型注解缺失
  - 依赖TensorFlow内部不稳定API
  - 多版本兼容性未明确
  - 性能基准数据缺失