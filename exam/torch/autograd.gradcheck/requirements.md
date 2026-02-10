# torch.autograd.gradcheck 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证数值梯度与解析梯度的一致性，使用有限差分法检查浮点/复数张量的梯度计算准确性
- 不在范围内的内容：不验证函数本身的数学正确性，仅验证梯度计算实现

## 2. 输入与约束
- 参数列表：
  - func: Callable，接收张量输入，返回张量或张量元组
  - inputs: Tensor/Sequence[Tensor]，需设置 requires_grad=True
  - eps: float=1e-6，有限差分扰动大小
  - atol: float=1e-5，绝对容差
  - rtol: float=1e-3，相对容差
  - raise_exception: bool=True，检查失败时是否抛出异常
  - check_sparse_nnz: bool=False，是否支持稀疏张量输入
  - nondet_tol: float=0.0，非确定性容差
  - check_undefined_grad: bool=True，检查未定义梯度处理
  - check_grad_dtypes: bool=False，检查梯度数据类型
  - check_batched_grad: bool=False，检查批处理梯度
  - check_batched_forward_grad: bool=False，检查批处理前向梯度
  - check_forward_ad: bool=False，检查前向模式自动微分
  - check_backward_ad: bool=True，检查后向模式自动微分
  - fast_mode: bool=False，快速模式（仅实函数）

- 有效取值范围/维度/设备要求：
  - 输入张量必须设置 requires_grad=True
  - 默认值针对双精度张量设计
  - 支持浮点/复数类型张量
  - 支持CPU和CUDA设备

- 必需与可选组合：
  - func 和 inputs 为必需参数
  - 其他参数均有默认值，可选择性配置

- 随机性/全局状态要求：
  - 函数本身应具有确定性
  - 有限差分法引入数值扰动

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 返回 bool 类型，所有差异满足 allclose 条件返回 True

- 容差/误差界（如浮点）：
  - 使用 torch.allclose 比较梯度
  - 默认容差：atol=1e-5, rtol=1e-3
  - 单精度张量可能检查失败

- 状态变化或副作用检查点：
  - 不应修改输入张量数据
  - 不应改变全局计算图状态

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 输入张量未设置 requires_grad=True
  - 函数返回非张量类型
  - 复数函数在 fast_mode=True 时
  - 重叠内存张量可能导致检查失败

- 边界值（空、None、0 长度、极端形状/数值）：
  - 空张量或零维度张量
  - 极端数值（inf, nan, 极大/极小值）
  - 单元素张量
  - 高维张量（>4维）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - 依赖 PyTorch 自动微分系统
  - 支持 CUDA 设备（如有）
  - 依赖 torch.allclose 实现

- 需要 mock/monkeypatch 的部分：
  - torch.allclose 调用
  - 自动微分计算路径
  - 有限差分计算

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基本实数函数梯度验证
  2. 复数函数 Wirtinger 导数检查
  3. 稀疏张量梯度检查（check_sparse_nnz=True）
  4. 前向模式自动微分验证（check_forward_ad=True）
  5. 异常处理：raise_exception=True/False 行为

- 可选路径（中/低优先级合并为一组列表）：
  - 批处理梯度检查（check_batched_grad=True）
  - 批处理前向梯度检查（check_batched_forward_grad=True）
  - 梯度数据类型检查（check_grad_dtypes=True）
  - 未定义梯度处理检查（check_undefined_grad=False）
  - 快速模式验证（fast_mode=True）
  - 不同精度张量（float16, float32, float64）
  - 复杂形状和维度组合
  - 多输出函数验证

- 已知风险/缺失信息（仅列条目，不展开）：
  - 复数函数梯度检查逻辑复杂
  - 快速模式仅支持实数到实数函数
  - 重叠内存张量行为未详细说明
  - 不同精度张量的具体容差要求不明确
  - 非确定性容差（nondet_tol）的具体应用场景