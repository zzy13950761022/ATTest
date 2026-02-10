# tensorflow.python.ops.functional_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 测试 foldl/foldr 左/右折叠操作的正确性
  - 验证 scan 扫描操作的累积行为
  - 确保 If 条件分支的正确执行
  - 测试 While/For 循环控制流
  - 验证多参数输入输出和嵌套结构支持
  - 确保 GPU-CPU 内存交换功能正常
  - 测试自动微分和梯度计算

- 不在范围内的内容
  - 底层 C++ 实现 (gen_functional_ops) 的内部逻辑
  - TensorFlow 框架本身的安装和配置
  - 非函数式操作模块的其他 TensorFlow 功能

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - fn: 可调用函数，接受 2+ 参数，返回张量或序列
  - elems: 张量或序列，第一维为序列长度，所有张量第一维必须匹配
  - initializer: 可选张量或序列，与 fn 返回值结构相同
  - parallel_iterations: 整数，默认 10，控制并行迭代数
  - back_prop: 布尔值，默认 True（已弃用）
  - swap_memory: 布尔值，默认 False，启用 GPU-CPU 内存交换
  - name: 字符串，默认 None，操作名称前缀
  - infer_shape: 布尔值，默认 True，仅 scan 函数
  - reverse: 布尔值，默认 False，仅 scan 函数
  - cond: 布尔张量，仅 If 函数
  - then_branch/else_branch: 可调用函数，仅 If 函数

- 有效取值范围/维度/设备要求
  - elems 张量第一维必须相同（序列长度）
  - 无 initializer 时 elems 必须至少包含一个元素
  - 支持嵌套结构（列表/元组）深度未明确限制
  - 支持 CPU 和 GPU 设备
  - 支持图模式和 eager 模式

- 必需与可选组合
  - fn 和 elems 为必需参数
  - initializer 为可选参数
  - 其他参数均有默认值

- 随机性/全局状态要求
  - 无随机性要求
  - 依赖 TensorFlow 全局图状态
  - 使用变量作用域缓存设备设置

## 3. 输出与判定
- 期望返回结构及关键字段
  - foldl/foldr: 返回与 fn 返回值相同结构的张量或序列
  - scan: 返回累积结果序列，形状为 `[len(values)] + fn(initializer, values[0]).shape`
  - If: 返回 then_branch 或 else_branch 的输出列表
  - While/For: 返回与输入相同类型的张量列表

- 容差/误差界（如浮点）
  - 浮点计算误差在 1e-6 范围内
  - 整数计算必须精确匹配
  - 形状推断必须准确

- 状态变化或副作用检查点
  - 验证变量作用域正确设置
  - 检查内存交换是否按预期触发
  - 验证梯度计算正确性
  - 确保无意外副作用

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - fn 不可调用时抛出 TypeError
  - elems 为空且无 initializer 时抛出 ValueError
  - elems 张量第一维不匹配时抛出 ValueError
  - 不支持的数据类型抛出 TypeError
  - back_prop=False 时发出弃用警告

- 边界值（空、None、0 长度、极端形状/数值）
  - 空张量输入（有 initializer）
  - 零维张量作为 elems
  - 极大序列长度（内存边界）
  - 极小/极大数值（浮点边界）
  - None 作为参数值
  - 深度嵌套结构（测试嵌套限制）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 库安装
  - GPU 设备（可选，用于 GPU 相关测试）
  - 足够内存处理大张量

- 需要 mock/monkeypatch 的部分
  - `tensorflow.python.ops.gen_functional_ops` 底层操作
  - `tensorflow.python.ops.control_flow_ops.while_loop` 循环实现
  - `tensorflow.python.framework.ops.get_default_graph` 图状态
  - `tensorflow.python.ops.variables.Variable` 变量操作
  - `tensorflow.python.ops.gradients_impl.gradients` 梯度计算

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. foldl/foldr 基本折叠操作正确性
  2. scan 累积序列生成验证
  3. If 条件分支正确执行
  4. While/For 循环控制流测试
  5. 嵌套结构和多参数支持

- 可选路径（中/低优先级合并为一组列表）
  - parallel_iterations 参数效果验证
  - swap_memory 内存交换功能测试
  - 图模式与 eager 模式行为一致性
  - 梯度计算和自动微分验证
  - 极端形状和大数据量性能测试
  - 不同设备（CPU/GPU）兼容性
  - 弃用参数 back_prop 的警告检查

- 已知风险/缺失信息（仅列条目，不展开）
  - 嵌套结构深度限制未文档化
  - parallel_iterations 并行实现细节未说明
  - swap_memory 具体触发条件未明确
  - 图模式与 eager 模式行为差异
  - 捕获输入（captured_inputs）处理逻辑复杂
  - 零维张量边界情况处理
  - 不同 dtype 混合输入支持程度