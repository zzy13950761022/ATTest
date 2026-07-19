# torch._linalg_utils 测试需求

## 1. 目标与范围
- 验证模块内线性代数工具函数的正确性
- 测试稀疏/密集矩阵混合运算、特征值计算、正交基生成
- 覆盖CPU与CUDA设备的实现差异
- 不包含：外部线性代数库验证、性能基准测试、数值稳定性分析

## 2. 输入与约束
- `matmul(A, B)`: A可为None/稀疏/密集张量，B总是密集张量
- `symeig(A, largest)`: A为对称矩阵，largest布尔值控制特征值排序
- `basis(A)`: A为矩阵(m×n)，返回列的正交基
- `bform(X, A, Y)`: X(m×k), A(m×n), Y(n×l)满足矩阵乘法维度
- `qform(A, S)`: A(m×n), S(n×k)满足矩阵乘法维度
- 设备要求：支持CPU和CUDA（需测试差异实现）
- 数据类型：浮点类型(float32/float64)，整数类型自动映射到float32

## 3. 输出与判定
- `matmul`: 返回Tensor，形状符合矩阵乘法规则
- `symeig`: 返回(特征值Tensor, 特征向量Tensor)，特征向量正交
- `basis`: 返回正交基矩阵，列向量正交且张成相同空间
- `bform/qform`: 返回计算结果Tensor，形状正确
- 容差：浮点误差在1e-6范围内
- 副作用：无全局状态修改，函数为纯计算

## 4. 错误与异常场景
- 非法输入：非张量参数触发TypeError
- 维度不匹配：矩阵乘法维度不兼容触发RuntimeError
- 非对称矩阵：symeig输入非对称矩阵触发未定义行为
- 空输入：空张量或零维度张量
- 极端形状：极大/极小矩阵尺寸
- 极端数值：极大/极小浮点值、NaN、Inf
- 已弃用函数：matrix_rank, solve, lstsq, eig触发RuntimeError

## 5. 依赖与环境
- 外部依赖：torch.sparse.mm, torch.matmul, torch.linalg.eigh, torch.linalg.qr
- 设备依赖：CUDA设备可用性影响basis函数实现
- 需要mock：无网络/文件依赖
- 需要monkeypatch：测试不同设备路径（CPU vs CUDA）

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. matmul稀疏与密集矩阵混合运算
  2. symeig对称矩阵特征值计算与排序
  3. basis函数CPU与CUDA实现差异
  4. conjugate函数复数与非复数类型处理
  5. 已弃用函数的RuntimeError验证

- 可选路径（中/低优先级）：
  - 不同浮点精度(float32/float64)测试
  - 整数类型自动映射到float32
  - 大规模矩阵性能边界测试
  - 随机矩阵生成与验证
  - 批处理张量支持测试

- 已知风险/缺失信息：
  - 部分函数缺少完整docstring
  - 未明确异常处理细节
  - 稀疏矩阵格式限制未说明
  - 内存使用边界未定义
  - 并发调用安全性未验证