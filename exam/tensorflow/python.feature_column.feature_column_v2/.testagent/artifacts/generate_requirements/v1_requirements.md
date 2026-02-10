# tensorflow.python.feature_column.feature_column_v2 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试 TensorFlow FeatureColumn API v2 模块的核心工厂函数，验证特征列创建、参数验证、类型转换和错误处理
- 不在范围内的内容：Estimator 集成、模型训练、分布式训练、自定义特征列实现

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - numeric_column: key(str), shape(tuple, default=(1,)), default_value(None), dtype(tf.dtype, default=tf.float32), normalizer_fn(callable/None)
  - categorical_column_with_vocabulary_list: key(str), vocabulary_list(list), dtype(tf.dtype/None), default_value(int, default=-1), num_oov_buckets(int, default=0)
  - bucketized_column: source_column(NumericColumn), boundaries(list)
  - embedding_column: categorical_column(CategoricalColumn), dimension(int), combiner(str, default='mean'), initializer(initializer/None), ckpt_to_load_from(str/None), tensor_name_in_ckpt(str/None), max_norm(float/None), trainable(bool, default=True)

- 有效取值范围/维度/设备要求：
  - key: 非空字符串，唯一标识符
  - shape: 正整数元组，支持标量(1,)和多维形状
  - vocabulary_list: 非空列表，元素类型一致（字符串或整数）
  - boundaries: 严格递增数值列表
  - dimension: 正整数嵌入维度
  - num_oov_buckets: 非负整数

- 必需与可选组合：
  - numeric_column: key必需，其他可选
  - categorical_column_with_vocabulary_list: key和vocabulary_list必需
  - default_value和num_oov_buckets互斥（不能同时设置）

- 随机性/全局状态要求：
  - embedding_column初始化为随机分布
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段：
  - numeric_column → NumericColumn对象，包含key, shape, dtype属性
  - categorical_column_with_vocabulary_list → CategoricalColumn对象，包含key, vocabulary_list属性
  - bucketized_column → BucketizedColumn对象，包含source_column, boundaries属性
  - embedding_column → EmbeddingColumn对象，包含categorical_column, dimension属性

- 容差/误差界（如浮点）：
  - 浮点边界值容差：1e-7
  - 形状转换无精度损失

- 状态变化或副作用检查点：
  - 无文件系统操作
  - 无网络请求
  - 无全局变量修改

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - key为空字符串或None → ValueError
  - vocabulary_list为空列表 → ValueError
  - boundaries非严格递增 → ValueError
  - dimension非正整数 → ValueError
  - shape包含非正整数 → ValueError
  - default_value和num_oov_buckets同时设置 → ValueError
  - 类型不匹配（如字符串传入数值列） → TypeError

- 边界值（空、None、0长度、极端形状/数值）：
  - shape=(0,) → ValueError
  - shape=(1000000,) → 内存检查
  - vocabulary_list长度极大 → 性能检查
  - boundaries包含inf/nan → ValueError
  - default_value超出词汇表索引范围 → 运行时错误

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow核心库
  - 无外部网络/文件依赖
  - 无GPU/TPU特定要求

- 需要mock/monkeypatch的部分：
  - normalizer_fn函数调用验证
  - embedding_column初始化的随机性
  - 实验性API（前缀"_"）的稳定性检查

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. numeric_column基础创建和参数验证
  2. categorical_column_with_vocabulary_list词汇表处理
  3. bucketized_column边界值分桶逻辑
  4. embedding_column维度验证和初始化
  5. 错误处理：无效key、空词汇表、非法形状

- 可选路径（中/低优先级合并为一组列表）：
  - 复杂形状支持（多维张量）
  - normalizer_fn函数集成
  - 词汇表外值处理策略
  - 特征列序列化和反序列化
  - 实验性API功能验证
  - 性能基准测试（大词汇表、多边界）

- 已知风险/缺失信息（仅列条目，不展开）：
  - 类型注解不完整
  - 部分错误消息格式不一致
  - 实验性API可能变更
  - 张量形状兼容性边界模糊
  - 默认值处理逻辑复杂