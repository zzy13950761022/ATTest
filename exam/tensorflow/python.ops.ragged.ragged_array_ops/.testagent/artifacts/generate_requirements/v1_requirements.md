# tensorflow.python.ops.ragged.ragged_array_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试 RaggedTensor 数组操作模块，包括 boolean_mask、tile、expand_dims、size、rank、reverse、cross、dynamic_partition、split、reshape 等函数对不规则张量的正确处理
- 不在范围内的内容：其他 TensorFlow 模块的 RaggedTensor 支持、非数组操作函数、性能基准测试

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - boolean_mask: data (Tensor/RaggedTensor), mask (Tensor/RaggedTensor), name (str/None)
  - tile: input (RaggedTensor), multiples (1-D Tensor), name (str/None)
  - expand_dims: input (RaggedTensor), axis (int/Tensor), name (str/None)
  - size: input (RaggedTensor), out_type (tf.dtype), name (str/None)
  - rank: input (RaggedTensor), name (str/None)
- 有效取值范围/维度/设备要求：
  - mask 的秩必须静态已知
  - mask.shape 必须是 data.shape 的前缀
  - multiples 长度必须与 input 维度数相同
  - 支持 RaggedTensor 和普通 Tensor 的混合操作
- 必需与可选组合：
  - boolean_mask: data 和 mask 必需，name 可选
  - tile: input 和 multiples 必需，name 可选
  - expand_dims: input 和 axis 必需，name 可选
- 随机性/全局状态要求：无全局状态依赖，函数为纯操作

## 3. 输出与判定
- 期望返回结构及关键字段：
  - boolean_mask: 返回与输入相同秩的潜在 RaggedTensor
  - tile: 返回与输入相同类型、秩和 ragged_rank 的 RaggedTensor
  - expand_dims: 返回在指定轴添加维度大小为1的张量
  - size: 返回标量张量表示元素总数
  - rank: 返回标量张量表示秩
- 容差/误差界（如浮点）：无特殊浮点容差要求
- 状态变化或副作用检查点：无副作用，不修改输入张量

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - mask 形状不是 data 形状的前缀
  - mask 秩未知（非静态）
  - multiples 长度与 input 维度数不匹配
  - 无效的 axis 值（超出范围）
  - 类型不匹配（如非布尔 mask）
- 边界值（空、None、0 长度、极端形状/数值）：
  - 空 RaggedTensor 输入
  - 标量输入处理
  - 全 True/False mask
  - 零维度 RaggedTensor
  - 极端嵌套深度

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：仅依赖 TensorFlow 运行时
- 需要 mock/monkeypatch 的部分：
  - `tensorflow.python.ops.ragged.ragged_tensor.RaggedTensor` 的转换方法
  - `tensorflow.python.ops.ragged.ragged_tensor.convert_to_tensor_or_ragged_tensor`
  - `tensorflow.python.ops.ragged.ragged_tensor.is_ragged`
  - `tensorflow.python.ops.ragged.ragged_tensor.RaggedTensor.from_nested_row_splits`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. boolean_mask 基本功能：正确过滤 RaggedTensor 元素
  2. tile 复制功能：验证多维复制逻辑
  3. expand_dims 维度扩展：单轴和多轴扩展
  4. size 和 rank 计算：验证元素计数和维度数
  5. 混合类型操作：RaggedTensor 与普通 Tensor 互操作
- 可选路径（中/低优先级合并为一组列表）：
  - reverse 反转操作
  - cross 向量叉积
  - dynamic_partition 动态分区
  - split 张量分割
  - reshape 形状重塑
  - 深度嵌套 RaggedTensor
  - 大规模数据性能
  - GPU 设备兼容性
- 已知风险/缺失信息（仅列条目，不展开）：
  - 部分函数缺少完整类型注解
  - 递归处理逻辑的深度限制
  - 内存使用边界情况
  - 稀疏张量互操作性
  - 分布式环境行为