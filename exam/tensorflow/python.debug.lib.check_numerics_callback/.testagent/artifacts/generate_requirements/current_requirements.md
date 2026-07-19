# tensorflow.python.debug.lib.check_numerics_callback 测试需求

## 1. 目标与范围
- 主要功能与期望行为：启用数值检查机制，在 TensorFlow eager/graph 执行中检测浮点张量的 NaN/Infinity 值，检测到异常时抛出 `tf.errors.InvalidArgumentError`
- 不在范围内的内容：非浮点数据类型的数值检查、性能开销量化、非 TensorFlow 环境

## 2. 输入与约束
- 参数列表：
  - stack_height_limit: int, 默认值 30，仅对 tf.function 操作生效
  - path_length_limit: int, 默认值 50，仅对 tf.function 操作生效
- 有效取值范围/维度/设备要求：
  - 参数必须为正整数
  - 仅对浮点数据类型 (dtype.is_floating) 进行检查
  - TPU 环境需先调用 `tf.config.set_soft_device_placement(True)`
- 必需与可选组合：两个参数均为可选，使用默认值
- 随机性/全局状态要求：线程局部状态，幂等性（多次调用效果相同）

## 3. 输出与判定
- 期望返回结构及关键字段：无返回值 (None)
- 容差/误差界（如浮点）：不适用
- 状态变化或副作用检查点：
  - 注册操作回调成功
  - 线程局部状态中记录回调实例
  - 监控计数器增加
  - 日志记录启用状态

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非整数参数类型
  - 零或负整数参数
  - TPU 环境未设置软设备放置
- 边界值（空、None、0 长度、极端形状/数值）：
  - 参数值为 0 或负数
  - 极大参数值（可能的内存/性能影响）
  - 浮点张量包含 NaN/Infinity 的边界情况

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow 运行时环境
  - TPU 设备（可选）
  - 线程局部存储
- 需要 mock/monkeypatch 的部分：
  - op_callbacks.add_op_callback
  - 线程局部状态管理
  - 日志记录系统
  - TPU 环境检测

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基本启用功能验证浮点张量 NaN/Infinity 检测
  2. 幂等性测试（多次调用 enable_check_numerics）
  3. 线程局部行为验证
  4. 参数边界值测试（stack_height_limit, path_length_limit）
  5. TPU 环境特殊要求验证
- 可选路径（中/低优先级合并为一组列表）：
  - 非浮点数据类型忽略测试
  - IGNORE_OP_OUTPUTS 列表覆盖验证
  - SAFE_OPS 列表跳过验证
  - disable_check_numerics 功能测试
  - 与 tf.function 集成测试
  - 极端形状/数值边界测试
  - 性能基准测试（可选）
- 已知风险/缺失信息（仅列条目，不展开）：
  - 非浮点数据类型处理不明确
  - 性能开销未量化
  - IGNORE_OP_OUTPUTS/SAFE_OPS 列表完整性
  - 并发环境下的线程安全性
  - 内存泄漏风险