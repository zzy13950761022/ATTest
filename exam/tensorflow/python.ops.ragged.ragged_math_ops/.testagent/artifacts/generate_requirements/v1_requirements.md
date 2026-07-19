# tensorflow.python.ops.ragged.ragged_math_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试 RaggedTensor 数学运算模块，验证 range、segment_*、reduce_*、matmul、softmax、add_n、dropout 等函数对不规则张量的正确支持
- 不在范围内的内容：标准 Tensor 的数学运算、非数学操作（如 reshape、concat）、第三方库集成

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - RaggedTensor：不规则张量，允许不同长度的维度
  - Tensor：标准张量，支持标量或向量
  - axis：归约轴，支持 None、int、list/tuple
  - dtype：int32、int64、float32、float64 等
  - segment_ids：int32 或 int64 类型
  - row_splits_dtype：tf.int32 或 tf.int64
- 有效取值范围/维度/设备要求：
  - 向量输入必须具有相同大小
  - 标量输入支持广播
  - 支持 CPU/GPU 设备
- 必需与可选组合：
  - range 函数：start、limit、delta 参数组合
  - reduce_* 函数：axis 参数可选
- 随机性/全局状态要求：
  - dropout 函数涉及随机性，需控制随机种子
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 大多数函数返回 RaggedTensor
  - 保持输入数据类型
  - 形状根据操作变化
- 容差/误差界（如浮点）：
  - 浮点运算误差在 1e-6 范围内
  - 整数运算精确匹配
- 状态变化或副作用检查点：
  - 无副作用，纯函数
  - 不修改输入张量

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 无效 segment_ids 类型（非 int32/int64）
  - 不兼容的广播维度
  - 无效的 axis 值（超出维度范围）
- 边界值（空、None、0 长度、极端形状/数值）：
  - 空 RaggedTensor 处理
  - 零长度维度
  - 负增量（range 函数）
  - 大尺寸 RaggedTensor（内存边界）
  - 极端数值（inf、nan、极大/极小值）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow 运行时环境
  - GPU 设备（可选）
  - 无网络/文件依赖
- 需要 mock/monkeypatch 的部分：
  - `tensorflow.python.ops.gen_ragged_math_ops`（底层 C++ 操作）
  - `tensorflow.python.ops.math_ops`（标准数学运算）
  - `tensorflow.python.ops.array_ops`（数组操作）
  - `tensorflow.python.ops.ragged.ragged_tensor`（RaggedTensor 类型）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. range 函数：空列表处理与 Python range 一致
  2. reduce_sum：多轴归约的顺序正确性
  3. segment_sum：segment_ids 有效性验证
  4. matmul：RaggedTensor 与标准 Tensor 混合运算
  5. dropout：随机性控制与期望值验证
- 可选路径（中/低优先级合并为一组列表）：
  - softmax 数值稳定性
  - add_n 多张量求和
  - 广播规则边界情况
  - 不同 dtype 组合兼容性
  - 性能基准测试（大尺寸处理）
- 已知风险/缺失信息（仅列条目，不展开）：
  - 未提供所有函数的完整类型注解
  - 多轴归约的顺序敏感性未明确说明
  - 大尺寸 RaggedTensor 的性能边界
  - 空列表处理与 tf.range 的差异