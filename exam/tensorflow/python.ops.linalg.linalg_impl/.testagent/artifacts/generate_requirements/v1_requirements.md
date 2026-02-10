# tensorflow.python.ops.linalg.linalg_impl 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证线性代数模块核心函数（logdet、adjoint、matrix_exponential、banded_triangular_solve、tridiagonal_solve、pinv、lu_solve等）的正确性、数值稳定性和异常处理
- 不在范围内的内容：底层C++实现细节、GPU/TPU特定优化、性能基准测试、第三方库集成

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：各函数独立，主要参数为输入张量，形状通常为`[..., M, M]`方阵
- 有效取值范围/维度/设备要求：支持float16, float32, float64, complex64, complex128数据类型；矩阵维度≥2；CPU/GPU设备兼容
- 必需与可选组合：输入张量为必需参数，部分函数支持可选参数如name、validate_args等
- 随机性/全局状态要求：无全局状态依赖；随机性仅用于测试数据生成

## 3. 输出与判定
- 期望返回结构及关键字段：返回Tensor类型，形状与输入匹配或按函数约定变换
- 容差/误差界（如浮点）：float32使用1e-5相对误差，float64使用1e-10相对误差；复数类型分别验证实部和虚部
- 状态变化或副作用检查点：无副作用；不修改输入张量；不改变全局状态

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：非方阵、奇异矩阵、非正定矩阵、不兼容数据类型、非法形状
- 边界值（空、None、0长度、极端形状/数值）：零矩阵、单位矩阵、极大/极小特征值矩阵、条件数极端矩阵

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：TensorFlow核心库；无网络/文件依赖
- 需要mock/monkeypatch的部分：`tensorflow.python.ops.gen_linalg_ops`底层操作；`tensorflow.python.ops.linalg.linalg_ops`辅助函数；`tensorflow.python.ops.array_ops`数组操作；`tensorflow.python.ops.math_ops`数学操作

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. logdet对Hermitian正定矩阵的正确性验证
  2. matrix_exponential数值稳定性边界测试
  3. tridiagonal_solve三种输入格式兼容性
  4. pinv对奇异矩阵的Moore-Penrose伪逆计算
  5. 复数数据类型在所有函数中的一致性

- 可选路径（中/低优先级合并为一组列表）：
  - 批量处理高维张量广播行为
  - 混合精度计算（float16与float32转换）
  - 极端条件数矩阵的数值稳定性
  - 非标准形状（如1x1矩阵）的特殊处理
  - 各函数梯度计算正确性
  - 设备间（CPU/GPU）计算结果一致性

- 已知风险/缺失信息（仅列条目，不展开）：
  - eigh_tridiagonal复数支持不完整
  - 高维张量广播行为文档不足
  - 部分函数数值稳定性边界未明确
  - 批量处理性能特性未定义
  - 异常消息格式未标准化