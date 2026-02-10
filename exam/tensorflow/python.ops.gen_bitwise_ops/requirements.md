# tensorflow.python.ops.gen_bitwise_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证位运算模块中所有函数（bitwise_and/or/xor/invert/left_shift/right_shift/population_count）的正确性，包括类型兼容性、形状广播、边界值处理
- 不在范围内的内容：非位运算操作、浮点数运算、自定义梯度计算、分布式执行

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - x, y: Tensor[int8/int16/int32/int64/uint8/uint16/uint32/uint64]，无默认值
  - name: str/None，默认None
- 有效取值范围/维度/设备要求：
  - 输入张量必须相同数据类型
  - 支持任意形状，自动广播
  - 支持CPU/GPU设备
- 必需与可选组合：
  - x, y必需，name可选
  - population_count仅需x参数
- 随机性/全局状态要求：无随机性，无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 返回与输入相同类型的Tensor（population_count返回uint8）
  - 形状符合广播规则
- 容差/误差界（如浮点）：精确位运算，无容差要求
- 状态变化或副作用检查点：无状态变化，纯函数

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 类型不匹配：x,y数据类型不同
  - 不支持类型：浮点、字符串等
  - 形状不兼容且无法广播
- 边界值（空、None、0长度、极端形状/数值）：
  - 空张量（shape=0）
  - 标量输入
  - 大形状张量（内存边界）
  - 移位操作：y负数、y≥位宽

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow运行时环境
  - 可选GPU支持
- 需要mock/monkeypatch的部分：
  - `tensorflow.python.framework.ops.device`（设备上下文）
  - `tensorflow.python.eager.context.executing_eagerly`（执行模式）
  - `tensorflow.python.ops.gen_bitwise_ops._op_def_library`（操作定义）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 所有支持数据类型的基本位运算正确性
  2. 形状广播功能验证
  3. eager和graph两种执行模式一致性
  4. 移位操作的边界情况处理
  5. 有符号整数取反的正确性
- 可选路径（中/低优先级合并为一组列表）：
  - 大尺寸张量性能测试
  - 不同设备（CPU/GPU）结果一致性
  - 梯度计算（如支持）
  - 与numpy位运算结果对比
  - 内存使用监控
- 已知风险/缺失信息（仅列条目，不展开）：
  - 移位操作边界行为实现定义
  - 有符号整数取反的负数表示
  - 模块级而非函数级测试覆盖
  - 缺少类型注解的静态检查