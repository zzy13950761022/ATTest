# tensorflow.python.eager.def_function 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 测试 `tf.function` 将 Python 函数编译为 TensorFlow 图的能力
  - 验证返回的 GenericFunction 对象可正确调用并执行
  - 测试支持数据依赖控制流（if/for/while/break/continue/return）
  - 验证变量初始化和状态管理机制
  - 测试重跟踪逻辑和性能优化

- 不在范围内的内容
  - 不测试 TensorFlow 核心张量运算的正确性
  - 不测试 AutoGraph 转换的完整语法覆盖
  - 不测试 XLA 编译后端的具体实现

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - func: Callable/None, 默认 None
  - input_signature: Sequence[tf.TensorSpec]/None, 默认 None
  - autograph: bool, 默认 True
  - jit_compile: bool/None, 默认 None
  - experimental_implements: str/None, 默认 None
  - experimental_autograph_options: tuple/None, 默认 None
  - experimental_relax_shapes: bool, 默认 False
  - experimental_compile: bool/None, 默认 None
  - experimental_follow_type_hints: bool/None, 默认 None

- 有效取值范围/维度/设备要求
  - func 必须是可调用对象或 None
  - input_signature 必须是 TensorSpec 序列或 None
  - 布尔参数必须为 True/False/None
  - 字符串参数必须为有效标识符或 None

- 必需与可选组合
  - func 为 None 时返回装饰器函数
  - input_signature 可限制重跟踪次数
  - experimental_compile 是 jit_compile 的已弃用别名

- 随机性/全局状态要求
  - 变量只能在第一次调用时创建
  - Python 副作用仅在跟踪时执行一次
  - 闭包可包含 tf.Tensor 和 tf.Variable 对象

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回 GenericFunction 对象（当 func 不为 None）
  - 返回装饰器函数（当 func 为 None）
  - GenericFunction 应包含多个 ConcreteFunction

- 容差/误差界（如浮点）
  - 图执行结果应与 eager 模式结果数值一致
  - 浮点误差在标准 TensorFlow 容差范围内

- 状态变化或副作用检查点
  - 验证变量状态在多次调用间保持
  - 检查重跟踪次数符合预期
  - 验证副作用操作仅执行一次

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 非可调用对象作为 func 参数
  - 无效的 input_signature 格式
  - 类型不匹配的 TensorSpec
  - 已弃用参数 experimental_compile 的使用警告

- 边界值（空、None、0 长度、极端形状/数值）
  - func=None 的装饰器模式
  - input_signature=None 的无约束模式
  - 空序列作为 input_signature
  - 极端形状张量（0维、大维度）
  - 布尔参数边界值（True/False/None）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 运行时环境
  - 可能的 GPU/TPU 设备依赖（当启用 XLA）
  - Python 运行时和标准库

- 需要 mock/monkeypatch 的部分
  - AutoGraph 转换过程（测试 autograph=False）
  - XLA 编译后端（测试 jit_compile 选项）
  - 变量初始化器
  - 重跟踪计数器

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 基本装饰器用法：@tf.function 装饰普通函数
  2. 带 input_signature 限制重跟踪
  3. 变量创建和状态保持验证
  4. 控制流语句（if/for/while）支持
  5. 重跟踪触发条件和次数验证

- 可选路径（中/低优先级合并为一组列表）
  - XLA 编译模式（jit_compile=True）
  - AutoGraph 禁用模式（autograph=False）
  - 类型注解优化（experimental_follow_type_hints）
  - 形状放宽选项（experimental_relax_shapes）
  - 已知函数实现（experimental_implements）
  - 装饰器工厂模式（func=None）
  - 闭包变量和自由变量处理
  - 多 ConcreteFunction 管理

- 已知风险/缺失信息（仅列条目，不展开）
  - experimental_compile 已弃用但需向后兼容
  - 变量初始化限制的具体边界
  - 重跟踪性能影响量化
  - AutoGraph 转换的完整边界情况
  - XLA 编译兼容性问题
  - 类型注解支持的完整范围