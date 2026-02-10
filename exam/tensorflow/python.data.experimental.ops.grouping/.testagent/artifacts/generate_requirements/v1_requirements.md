# tensorflow.python.data.experimental.ops.grouping 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证 `group_by_reducer` 按键分组并应用归约操作
  - 验证 `group_by_window` 按窗口分组转换（已弃用）
  - 验证 `bucket_by_sequence_length` 按序列长度分桶和批处理（已弃用）
  - 确保返回的函数可通过 `tf.data.Dataset.apply` 正确应用
- 不在范围内的内容
  - 不测试已弃用函数的替代实现（`tf.data.Dataset.group_by_window` 等）
  - 不测试底层 TensorFlow 数据集核心功能
  - 不测试用户自定义函数的内部逻辑

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `group_by_reducer`: key_func(函数→tf.int64标量), reducer(Reducer实例)
  - `group_by_window`: key_func(函数→tf.int64标量), reduce_func(函数), window_size(tf.int64标量)或window_size_func(函数)
  - `bucket_by_sequence_length`: element_length_func(函数→tf.int32), bucket_boundaries(list<int>), bucket_batch_sizes(list<int>)
- 有效取值范围/维度/设备要求
  - key_func 必须返回标量 tf.int64 张量
  - bucket_batch_sizes 长度必须为 len(bucket_boundaries)+1
  - window_size 和 window_size_func 互斥
- 必需与可选组合
  - group_by_window: window_size 或 window_size_func 必须提供其一
  - bucket_by_sequence_length: bucket_boundaries 和 bucket_batch_sizes 必需
- 随机性/全局状态要求
  - 无随机性要求
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - 所有函数返回 `Dataset` 转换函数（可调用）
  - 返回函数接受数据集参数，返回转换后的数据集
  - 转换后数据集保持 TensorFlow 数据集接口
- 容差/误差界（如浮点）
  - 无浮点容差要求
  - 分组键匹配必须精确
- 状态变化或副作用检查点
  - 验证弃用警告触发（group_by_window, bucket_by_sequence_length）
  - 验证函数包装器正确创建（StructuredFunctionWrapper）

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - key_func 返回非 tf.int64 类型触发异常
  - key_func 返回非标量张量触发异常
  - bucket_batch_sizes 长度不匹配触发异常
  - window_size 和 window_size_func 同时提供触发异常
  - 无效的 bucket_boundaries（非递增列表）触发异常
- 边界值（空、None、0 长度、极端形状/数值）
  - 空数据集输入验证
  - bucket_boundaries 空列表边界
  - 零窗口大小边界
  - 极大序列长度边界
  - 负值边界检查

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 数据集 API
  - Python 函数包装机制
- 需要 mock/monkeypatch 的部分
  - 弃用警告捕获和验证
  - StructuredFunctionWrapper 行为验证
  - 用户自定义函数的调用跟踪

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. group_by_reducer 基本分组归约功能
  2. bucket_by_sequence_length 序列分桶和填充
  3. 参数验证异常路径（类型/维度错误）
  4. 空数据集和边界值处理
  5. 弃用警告正确触发

- 可选路径（中/低优先级合并为一组列表）
  - group_by_window 窗口分组功能
  - 复杂嵌套结构的 key_func
  - 多设备环境验证
  - 性能基准测试
  - 内存使用监控

- 已知风险/缺失信息（仅列条目，不展开）
  - Reducer 类使用示例缺失
  - group_by_reducer 完整示例缺失
  - 已弃用函数维护状态不明确
  - 函数包装器内部错误处理细节