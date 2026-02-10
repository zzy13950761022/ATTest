# tensorflow.python.ops.ragged.ragged_functional_ops 测试需求

## 1. 目标与范围
- 主要功能：对 RaggedTensor 的 flat_values 应用操作 op，保持 ragged 结构
- 期望行为：替换 args/kwargs 中的 RaggedTensor 为 flat_values，应用 op，用原 nested_row_splits 重建 RaggedTensor
- 不在范围：不测试 op 应用到 RaggedTensor 每一行（与 tf.map_fn 不同）

## 2. 输入与约束
- op (callable)：可调用对象，必须保持最外层维度大小不变
- *args：位置参数，可包含 RaggedTensor 或其他类型
- **kwargs：关键字参数，可包含 RaggedTensor 或其他类型
- 有效约束：
  - 多个 RaggedTensor 输入时，必须具有相同的 nested_row_splits
  - op 返回值的 shape[0] 必须匹配 RaggedTensor flat_values 的 shape[0]
  - partition dtypes 必须兼容或可自动转换
- 随机性/全局状态：无随机性或全局状态修改

## 3. 输出与判定
- 期望返回：RaggedTensor 对象
- ragged_rank：与所有输入 RaggedTensor 的 ragged_rank 匹配
- 无 RaggedTensor 输入时：直接返回 op(*args, **kwargs)
- 容差：浮点运算误差遵循 TensorFlow 标准容差
- 状态变化：无副作用，不修改输入参数

## 4. 错误与异常场景
- 非法输入：op 不是可调用对象
- 维度不匹配：多个 RaggedTensor 的 nested_row_splits 不同
- 形状违规：op 返回值 shape[0] 与 flat_values shape[0] 不匹配
- 边界值：
  - 空 RaggedTensor（如 `tf.ragged.constant([])`）
  - 全空行的 RaggedTensor（如 `tf.ragged.constant([[], []])`）
  - 单元素 RaggedTensor
  - 极端形状（超大 ragged_rank，超长 flat_values）
- 类型错误：partition dtypes 不兼容且无法自动转换

## 5. 依赖与环境
- 外部依赖：TensorFlow RaggedTensor 实现
- 需要 mock 的部分：
  - `tensorflow.python.ops.ragged.ragged_functional_ops._replace_ragged_with_flat_values`
  - `tensorflow.python.ops.ragged.ragged_functional_ops._merge_partition_lists`
  - `tensorflow.python.ops.ragged.ragged_tensor.RaggedTensor._from_nested_row_partitions`
- 设备要求：支持 CPU 和 GPU（如可用）

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. 单个 RaggedTensor 输入，简单 op（如 tf.ones_like）
  2. 多个 RaggedTensor 输入，相同 nested_row_splits
  3. 无 RaggedTensor 输入，直接调用 op
  4. RaggedTensor 在嵌套结构（列表/字典）中
  5. op 返回值 shape[0] 不匹配的错误处理
- 可选路径（中/低优先级）：
  - 混合类型参数（RaggedTensor + 普通张量 + 标量）
  - 不同 partition dtypes 的自动转换
  - 极端形状和边界值
  - 复杂 op 函数（多参数、关键字参数）
  - 性能测试（大规模数据）
- 已知风险/缺失信息：
  - op 参数的具体类型注解缺失
  - op 返回张量形状的完整约束说明不足
  - 嵌套数据结构深度限制未定义
  - 内存使用边界未说明