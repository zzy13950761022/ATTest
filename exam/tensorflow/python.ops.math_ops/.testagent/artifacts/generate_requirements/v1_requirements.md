# tensorflow.python.ops.math_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证数学运算模块中代表性函数（AddV2, segment_sum, reduce_mean等）的正确性，包括基本算术、分段操作、归约运算，确保支持多种数值类型和numpy风格广播
- 不在范围内的内容：不测试模块中所有100+个函数，不验证GPU特定优化，不覆盖已弃用函数

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - data: Tensor，支持float32/64, int32/64, uint8/16/32/64, complex64/128, bfloat16, half, qint8/32, quint8
  - segment_ids: Tensor[int32/int64]，1-D张量，大小等于data第一维，值应排序且可重复
  - name: string/None，操作名称（可选）
- 有效取值范围/维度/设备要求：支持CPU和GPU，segment_ids在CPU上验证排序，GPU上不验证
- 必需与可选组合：data必需，segment_ids必需（分段函数），name可选
- 随机性/全局状态要求：无随机性，无全局状态修改

## 3. 输出与判定
- 期望返回结构及关键字段：Tensor类型，与输入data类型相同，形状取决于分段结果
- 容差/误差界（如浮点）：浮点运算容差1e-6，复数运算验证实部虚部
- 状态变化或副作用检查点：无I/O操作，无全局状态修改，仅返回计算结果

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：非数值类型输入，segment_ids维度不匹配，未排序segment_ids（CPU）
- 边界值（空、None、0长度、极端形状/数值）：空分段输出为0，极端数值（inf, nan, 极大/小值），0长度张量

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：TensorFlow运行时，CPU/GPU设备可用性
- 需要mock/monkeypatch的部分：
  - `tensorflow.python.ops.gen_math_ops` 底层操作
  - `tensorflow.python.framework.ops.convert_to_tensor` 类型转换
  - `tensorflow.python.ops.array_ops` 形状处理
  - `tensorflow.python.eager.context` 执行上下文

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. AddV2基本算术运算和广播
  2. segment_sum分段求和正确性
  3. reduce_mean归约运算
  4. 多种数值类型支持验证
  5. 空输入和边界条件处理
- 可选路径（中/低优先级合并为一组列表）：
  - 复数运算边界条件
  - GPU特定行为差异
  - 已弃用函数兼容性
  - 极端形状广播验证
  - 性能基准测试
- 已知风险/缺失信息（仅列条目，不展开）：
  - GPU上segment_ids排序验证行为未明确
  - 完整函数列表未提供
  - 复数运算边界条件未详细说明
  - 数值精度和溢出处理需验证