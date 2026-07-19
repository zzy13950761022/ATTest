# tensorflow.python.ops.gen_linalg_ops 测试需求

## 1. 目标与范围
- 验证线性代数操作（Cholesky、QR、SVD、矩阵求逆、求解、特征值等）的正确性
- 确保批量操作、多种数值类型（浮点、复数）支持
- 验证形状约束、设备兼容性、梯度计算
- 不在范围：底层C++实现细节、性能基准测试、第三方库集成

## 2. 输入与约束
- 参数：输入张量（float64/float32/half/complex64/complex128）、可选布尔标志（adjoint/full_matrices/lower等）、name
- 形状约束：方阵 `[..., M, M]` 或一般矩阵 `[..., M, N]`，前导维度为批量维度
- 设备要求：CPU/GPU兼容，大矩阵GPU性能更优
- 必需组合：Cholesky输入必须对称正定，MatrixSolve需要可逆系数矩阵
- 随机性：无全局状态要求，操作应为确定性

## 3. 输出与判定
- 返回结构：单个张量（Cholesky、MatrixInverse）或多个张量（QR、SVD、Eig）
- 容差：浮点误差在1e-6内，复数误差需验证实部虚部分别
- 状态变化：无副作用，纯函数行为
- 检查点：输出形状匹配输入形状约束，数值正确性验证

## 4. 错误与异常场景
- 非法输入：非正定矩阵调用Cholesky，奇异矩阵求逆
- 维度错误：非方阵调用Cholesky/MatrixInverse，形状不匹配
- 类型错误：不支持的数据类型（如int32）
- 边界值：空张量、零维张量、零大小矩阵
- 极端形状：超大矩阵（内存限制）、极小矩阵（1x1）
- 极端数值：极大/极小浮点数、NaN、Inf

## 5. 依赖与环境
- 外部依赖：TensorFlow C++核心库（linalg_ops.cc）
- 设备依赖：CUDA GPU支持（可选）
- 需要mock：`pywrap_tfe.TFE_Py_FastPathExecute`（执行路径）
- 需要monkeypatch：`_execute.record_gradient`（梯度记录）
- 需要监控：`_core._NotOkStatusException`（C++错误传播）

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. Cholesky分解对称正定矩阵验证
  2. QR分解形状保持与正交性验证
  3. 矩阵求逆可逆性验证与逆矩阵性质
  4. SVD分解重构误差验证
  5. 批量操作跨批次一致性验证

- 可选路径（中/低优先级）：
  - 复数矩阵操作正确性
  - 不同数值类型（half/float32/float64）精度差异
  - GPU与CPU结果一致性
  - 梯度计算正确性（前向/反向传播）
  - 病态矩阵数值稳定性
  - 边缘形状（1x1, 2x2, 大尺寸）处理
  - 可选参数（adjoint, full_matrices, lower）组合

- 已知风险/缺失信息：
  - 不可逆矩阵行为未定义
  - 奇异矩阵处理方式未明确
  - 数值精度保证未详细说明
  - 部分函数文档缺失（如banded_triangular_solve）
  - 类型转换规则未明确
  - 形状验证详细规则未说明