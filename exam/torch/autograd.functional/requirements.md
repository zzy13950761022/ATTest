# torch.autograd.functional 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试高阶自动微分函数（vjp, jvp, jacobian, hessian, vhp, hvp）的正确性、性能和边界处理
- 不在范围内的内容：底层自动微分引擎实现、非Tensor输入转换、自定义梯度函数

## 2. 输入与约束
- 参数列表：
  - `func`: 可调用函数，输入Tensor/元组，输出Tensor/元组
  - `inputs`: Tensor或Tensor元组，支持任意形状
  - `v`: Tensor或Tensor元组（vjp/jvp/vhp/hvp），形状与输出/输入匹配
  - `create_graph`: bool，默认False，控制是否创建计算图
  - `strict`: bool，默认False，检测输入输出独立性
  - `vectorize`: bool（jacobian/hessian），默认False，实验性向量化
  - `strategy`: str（jacobian），"reverse-mode"或"forward-mode"
  - `outer_jacobian_strategy`: str（hessian），"reverse-mode"或"forward-mode"

- 有效取值范围/维度/设备要求：
  - 输入必须是Tensor或Tensor元组
  - 支持CPU/CUDA设备
  - 支持任意维度（0D标量到高维张量）
  - 浮点类型（float32/float64）

- 必需与可选组合：
  - `func`和`inputs`为必需参数
  - `v`在vjp/jvp/vhp/hvp中必需
  - `vectorize=True`时`strategy`必须为"forward-mode"
  - `strict=True`与`vectorize=True`不兼容
  - `create_graph=True`与正向模式策略不兼容

- 随机性/全局状态要求：无全局状态依赖，结果应确定（给定相同输入）

## 3. 输出与判定
- 期望返回结构及关键字段：
  - vjp/jvp/vhp/hvp：返回(output, result)元组
  - jacobian/hessian：返回微分矩阵/张量
  - 输出形状与输入输出维度匹配

- 容差/误差界（如浮点）：
  - 浮点误差：相对误差<1e-5，绝对误差<1e-7
  - 与数值微分对比验证
  - 梯度检查通过torch.autograd.grad验证

- 状态变化或副作用检查点：
  - `create_graph=False`时不保留计算图
  - 无全局状态修改
  - 内存使用应在合理范围内

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非Tensor输入触发TypeError
  - 函数返回非Tensor触发RuntimeError
  - 形状不匹配触发RuntimeError
  - 无效策略参数触发ValueError
  - 不兼容参数组合触发RuntimeError

- 边界值（空、None、0长度、极端形状/数值）：
  - 空Tensor输入（形状包含0）
  - 标量输入（0维Tensor）
  - 极大/极小数值（inf, nan, 极值）
  - 大维度张量（内存边界）
  - 嵌套元组深度边界

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - PyTorch库依赖
  - CUDA设备（可选）
  - 无网络/文件依赖

- 需要mock/monkeypatch的部分：
  - torch.autograd.grad调用
  - torch._vmap_internals._vmap（向量化）
  - 设备内存分配

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 基本vjp/jvp功能验证（标量/向量/矩阵）
  2. jacobian/hessian矩阵正确性（与数值微分对比）
  3. create_graph参数对计算图的影响
  4. strict模式检测独立输入输出
  5. 正向/反向模式策略功能验证

- 可选路径（中/低优先级合并为一组列表）：
  - 向量化功能测试（vectorize=True）
  - 极端形状和大规模张量性能
  - 混合精度计算（float16/float32/float64）
  - 复杂嵌套函数链式微分
  - 多设备（CPU/CUDA）一致性
  - 内存泄漏和性能基准

- 已知风险/缺失信息（仅列条目，不展开）：
  - vectorize参数标记为实验性
  - 正向模式AD要求vectorize=True
  - strict=True与vectorize=True不兼容
  - create_graph=True与正向模式不兼容
  - 缺少复杂形状边界处理指南