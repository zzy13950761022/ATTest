# tensorflow.python.framework.constant_op 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证 `constant()` 和 `constant_v1()` 正确创建常量张量
  - 测试从 Python 标量、列表、numpy 数组到张量的转换
  - 验证形状重塑和广播功能
  - 确保数据类型推断和显式指定正常工作
- 不在范围内的内容
  - 符号张量（symbolic tensors）处理
  - 分布式环境下的设备放置策略
  - 性能基准测试和内存优化

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `value`: 任意类型，无默认值，支持标量/列表/数组
  - `dtype`: tf.DType/None，默认 None，从 value 推断
  - `shape`: 列表/元组/None，默认 None，使用 value 形状
  - `name`: 字符串，默认 "Const"
  - `verify_shape`: 布尔值，仅 constant_v1，默认 False
- 有效取值范围/维度/设备要求
  - value 支持 Python 原生类型和 numpy 数组
  - shape 必须与 value 兼容（广播或重塑）
  - 在当前设备（CPU/GPU）上创建张量
- 必需与可选组合
  - value 必需，其他参数可选
  - dtype 和 shape 可同时指定或单独指定
- 随机性/全局状态要求
  - 无随机性要求
  - eager 模式下可能使用缓存机制

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回 tf.Tensor 类型（eager 模式为 EagerTensor）
  - 张量值必须与输入 value 一致
  - 数据类型必须匹配指定或推断的 dtype
  - 形状必须匹配指定或推断的 shape
- 容差/误差界（如浮点）
  - 浮点数比较使用相对容差 1e-6
  - 整数类型必须精确匹配
- 状态变化或副作用检查点
  - 验证张量创建不修改输入数据
  - 检查 eager 模式下的缓存行为

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 无效 shape 参数（负值、非整数）触发 ValueError
  - 不兼容的 shape 和 value 触发 ValueError
  - 无效 dtype 触发 TypeError
  - 符号张量输入触发 TypeError
- 边界值（空、None、0 长度、极端形状/数值）
  - 空列表 [] 创建形状为 (0,) 的张量
  - 标量 0 创建形状为 () 的张量
  - 极端形状（大维度）测试内存限制
  - 数值边界（inf, nan, 极大/极小值）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 运行时环境
  - numpy 库（用于数组输入）
  - CPU/GPU 设备可用性
- 需要 mock/monkeypatch 的部分
  - eager/graph 模式切换
  - 设备放置策略
  - 缓存机制验证

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 基本标量和列表创建常量张量
  2. dtype 显式指定和自动推断
  3. shape 重塑和广播功能
  4. constant_v1 的 verify_shape 参数验证
  5. numpy 数组输入转换
- 可选路径（中/低优先级合并为一组列表）
  - 嵌套列表和复杂数据结构
  - 大尺寸张量创建性能
  - 特殊数据类型（复数、字符串）
  - 设备间张量复制行为
  - 缓存机制详细验证
- 已知风险/缺失信息（仅列条目，不展开）
  - 符号张量处理未明确
  - 广播规则的详细约束
  - 缓存机制的具体实现
  - 设备放置的默认策略