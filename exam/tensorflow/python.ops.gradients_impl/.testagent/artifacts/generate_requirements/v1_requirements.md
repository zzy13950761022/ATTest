# tensorflow.python.ops.gradients_impl 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 计算张量 `ys` 对 `xs` 的符号导数（梯度）
  - 返回长度为 `len(xs)` 的张量列表，每个张量为 `sum(dy/dx)`
  - 支持偏导数计算（通过 `stop_gradients`）
  - 处理未连接梯度的不同策略
- 不在范围内的内容
  - Eager 模式下的梯度计算（仅在图上下文中有效）
  - 高阶导数（Hessians）计算（由其他函数处理）
  - 自动微分引擎的内部实现细节

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `ys`: Tensor/list[Tensor]，需要微分的张量或列表
  - `xs`: Tensor/list[Tensor]，用于微分的张量或列表
  - `grad_ys`: Tensor/list[Tensor]/None，默认 None（使用全1张量）
  - `name`: str，默认 "gradients"
  - `colocate_gradients_with_ops`: bool，默认 False
  - `gate_gradients`: bool，默认 False
  - `aggregation_method`: AggregationMethod/None，默认 None
  - `stop_gradients`: Tensor/list[Tensor]/None，默认 None
  - `unconnected_gradients`: UnconnectedGradients，默认 NONE
- 有效取值范围/维度/设备要求
  - `ys` 和 `xs` 必须为同一图中的张量
  - `grad_ys` 必须与 `ys` 形状相同（当提供时）
  - 整数张量自动视为常数（相当于包含在 `stop_gradients` 中）
- 必需与可选组合
  - `ys` 和 `xs` 为必需参数
  - 其他参数均为可选，有默认值
- 随机性/全局状态要求
  - 无随机性要求
  - 需要图上下文（非 Eager 模式）

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回长度为 `len(xs)` 的张量列表
  - 每个张量为 `sum(dy/dx)` 对 y∈ys 和 x∈xs
  - 可能包含 None（当 `unconnected_gradients='none'` 且未连接时）
- 容差/误差界（如浮点）
  - 浮点计算容差：相对误差 1e-6，绝对误差 1e-8
  - 零梯度检查：梯度值小于 1e-12 视为零
- 状态变化或副作用检查点
  - 图操作修改：添加梯度计算节点
  - 控制流梯度图创建受图变异锁保护

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - `ys` 或 `xs` 为空列表或 None
  - `grad_ys` 形状与 `ys` 不匹配
  - 非张量类型输入
  - 不同图中的张量
  - 无效的 `unconnected_gradients` 枚举值
- 边界值（空、None、0 长度、极端形状/数值）
  - 单元素列表与单个张量的等价性
  - 零形状张量
  - 极端数值（inf, nan, 极大/极小值）
  - 整数张量（应视为常数）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 图执行环境
  - 无外部网络或文件依赖
- 需要 mock/monkeypatch 的部分
  - `tensorflow.python.ops.gradients_util._GradientsHelper`
  - `tensorflow.python.framework.ops.get_default_graph`
  - `tensorflow.python.ops.array_grad` 等梯度模块
  - `tensorflow.python.framework.ops._control_flow_state`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 基本梯度计算：单张量 ys 对单张量 xs
  2. 列表输入：多 ys 对多 xs 的梯度聚合
  3. 偏导数计算：使用 stop_gradients 参数
  4. 未连接梯度处理：none 和 zero 策略
  5. 自定义初始梯度：通过 grad_ys 参数
- 可选路径（中/低优先级合并为一组列表）
  - 布尔参数组合测试：colocate_gradients_with_ops 和 gate_gradients
  - 不同 aggregation_method 的影响
  - 控制流操作（while_loop, cond）的梯度计算
  - 整数张量自动视为常数的行为
  - 名称作用域和操作命名
- 已知风险/缺失信息（仅列条目，不展开）
  - 多实体情况：模块包含多个函数（gradients, gradients_v2, hessians 等）
  - 类型信息缺失：参数的具体类型约束不够明确
  - 性能影响：gate_gradients 和 colocate_gradients_with_ops 的实际效果不明确
  - 依赖关系：深度依赖 gradients_util._GradientsHelper 的内部实现