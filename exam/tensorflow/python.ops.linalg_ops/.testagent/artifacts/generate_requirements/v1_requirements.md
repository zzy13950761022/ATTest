# tensorflow.python.ops.linalg_ops 测试需求

## 1. 目标与范围
- 验证线性代数核心函数（matrix_triangular_solve, matrix_solve_ls, svd, norm, eig, cholesky_solve）的正确性
- 测试批量处理、数据类型兼容性、数值稳定性
- 不在范围内：性能基准测试、GPU/TPU特定优化、第三方库集成

## 2. 输入与约束
- 参数：各函数独立，包含矩阵输入、算法选项、正则化参数
- 数据类型：float32, float64, half, complex64, complex128
- 形状约束：矩阵维度兼容性（如方阵、可逆性）
- 设备要求：CPU/GPU支持，无特殊网络依赖
- 必需组合：matrix_triangular_solve需要三角矩阵输入
- 随机性：无全局状态，但算法可能有随机初始化

## 3. 输出与判定
- 返回结构：Tensor或元组（如SVD返回U,S,V）
- 容差：浮点误差<1e-6，复数误差<1e-8
- 形状验证：输出维度与输入广播规则一致
- 副作用检查：无文件/网络操作，无全局状态修改

## 4. 错误与异常场景
- 非法输入：非数值类型、无效形状、不可逆矩阵
- 维度错误：非方阵求逆、形状不匹配的矩阵乘法
- 类型错误：不支持的数据类型组合
- 边界值：空张量、零维度、极端数值（inf/nan）
- 特殊限制：complex128与l2_regularizer不兼容

## 5. 依赖与环境
- 外部依赖：TensorFlow核心库，无网络/文件依赖
- 需要mock：无外部服务调用
- 需要monkeypatch：无
- 设备依赖：支持CPU，可选GPU测试

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. matrix_triangular_solve基本三角求解
  2. svd分解正确性与奇异值排序
  3. 批量处理不同形状矩阵
  4. 数据类型兼容性（float32/64, complex64/128）
  5. 错误处理（无效输入、形状不匹配）

- 可选路径（中/低优先级）：
  - fast/slow算法路径选择
  - 数值稳定性边界条件
  - 复数运算特殊处理
  - 大规模矩阵性能
  - 与NumPy结果对比

- 已知风险/缺失信息：
  - 缺少__all__定义，API边界模糊
  - fast=True条件限制未文档化
  - complex128与l2_regularizer禁用原因
  - 批量维度广播规则细节