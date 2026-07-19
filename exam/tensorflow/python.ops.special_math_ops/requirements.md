# tensorflow.python.ops.special_math_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 特殊数学函数计算：贝塞尔函数、Dawson积分、Fresnel积分、指数积分等
  - 张量收缩运算：einsum 支持多种张量操作
  - 对数 Beta 函数计算：lbeta 沿最后维度缩减
  - 数值稳定性：提供修改的贝塞尔函数（i0e, i1e, k0e, k1e）
  - 数据类型支持：float32/float64/half，SparseTensor 兼容性

- 不在范围内的内容
  - 通用数学运算（在 math_ops 中）
  - 线性代数运算（在 linalg_ops 中）
  - 性能基准测试和优化策略评估
  - 与第三方库（如 SciPy）的完全一致性验证

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - x: Tensor/SparseTensor，支持 float32/float64/half
  - name: str/None，操作名称，可选
  - equation: str，einsum 方程字符串
  - *inputs: 可变参数，要收缩的输入张量
  - optimize: str，einsum 优化策略（greedy/optimal/branch-2/branch-all/auto）

- 有效取值范围/维度/设备要求
  - 特殊函数：实数域，部分函数有定义域限制
  - einsum：方程语法需符合 Einstein 求和约定
  - 维度兼容：广播规则适用
  - 设备：CPU/GPU 支持

- 必需与可选组合
  - 特殊函数：x 必需，name 可选
  - einsum：equation 和至少一个输入必需，optimize 可选
  - lbeta：至少一个输入张量必需

- 随机性/全局状态要求
  - 无全局状态依赖
  - 无随机性操作
  - 确定性计算

## 3. 输出与判定
- 期望返回结构及关键字段
  - 特殊函数：与输入相同类型和形状的 Tensor/SparseTensor
  - einsum：根据方程确定的收缩后 Tensor
  - lbeta：沿最后一个维度缩减的对数 Beta 值 Tensor

- 容差/误差界（如浮点）
  - float32：相对误差 1e-5，绝对误差 1e-6
  - float64：相对误差 1e-10，绝对误差 1e-12
  - 边界值：大/小输入值的数值稳定性
  - 特殊点：0, ±inf, NaN 的处理

- 状态变化或副作用检查点
  - 无文件 I/O 操作
  - 无网络请求
  - 无全局变量修改
  - 无副作用函数调用

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 无效数据类型：非数值类型输入
  - 维度不匹配：einsum 方程与输入形状冲突
  - 定义域外：特殊函数的无效输入值
  - 空方程：einsum 空字符串
  - 无效优化策略：optimize 参数非法值

- 边界值（空、None、0 长度、极端形状/数值）
  - 空张量：lbeta 对空维度返回 -inf
  - 零值：特殊函数在 0 点的行为
  - 极大值：数值溢出/下溢处理
  - 极小值：接近 0 的数值稳定性
  - 负值：定义域检查
  - 极端形状：高维张量（>5维）处理

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - Cephes 数学库（底层实现）
  - opt_einsum（einsum 优化）
  - 无网络/文件系统依赖
  - CPU/GPU 计算资源

- 需要 mock/monkeypatch 的部分
  - `tensorflow.python.ops.gen_special_math_ops`（核心操作）
  - `tensorflow.python.ops.gen_linalg_ops`（线性代数操作）
  - `tensorflow.python.ops.math_ops`（数学运算）
  - `tensorflow.python.ops.array_ops`（数组操作）
  - `opt_einsum.contract`（einsum 优化实现）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 特殊函数基本功能：贝塞尔函数、Dawson积分、Fresnel积分的标准输入
  2. einsum 核心操作：矩阵乘法、点积、外积、转置的标准用例
  3. 数据类型兼容性：float32/float64/half 在所有函数中的一致性
  4. 边界值处理：0, inf, NaN, 空张量的正确响应
  5. 数值稳定性：修改的贝塞尔函数与标准版本的对比

- 可选路径（中/低优先级合并为一组列表）
  - SparseTensor 支持验证
  - einsum 优化策略（greedy/optimal/branch-2/branch-all/auto）效果
  - 高维张量（>5维）的特殊函数计算
  - 与 SciPy 实现的数值一致性（参考值）
  - 批量处理性能：多批次输入的一致性
  - 梯度计算：特殊函数的自动微分验证
  - 混合精度计算：float16 与 float32 的转换

- 已知风险/缺失信息（仅列条目，不展开）
  - 部分特殊函数的数值稳定性边界未明确
  - SparseTensor 支持范围未完全文档化
  - einsum 优化策略的默认行为可能变化
  - 特殊函数在极端输入值的误差界未量化
  - 与 Cephes 库的版本兼容性依赖