# tensorflow.python.data.experimental.ops.counter 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证 CounterV2 函数创建无限计数数据集，从 start 开始按 step 步长生成序列，返回正确数据类型的 Dataset 对象
- 不在范围内的内容：CounterV1 和 Counter 函数的兼容性测试、数据集操作性能基准测试、分布式环境下的行为

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - start: int/float，标量，默认 0
  - step: int/float，标量，默认 1  
  - dtype: tf.dtype，默认 tf.int64
- 有效取值范围/维度/设备要求：start 和 step 支持正负数值，dtype 需为 TensorFlow 支持的数值类型
- 必需与可选组合：所有参数均为可选，使用默认值组合
- 随机性/全局状态要求：无随机性，确定性输出，不依赖全局状态

## 3. 输出与判定
- 期望返回结构及关键字段：返回 Dataset 对象，元素为标量 TensorSpec(shape=(), dtype=dtype)
- 容差/误差界（如浮点）：整数类型精确匹配，浮点类型允许数值误差
- 状态变化或副作用检查点：无副作用，不修改外部状态，每次调用创建独立数据集

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：无效 dtype 参数、非数值 start/step 参数、不支持的数据类型
- 边界值（空、None、0 长度、极端形状/数值）：start/step 为 0、极大/极小数值、浮点步长、负步长递减

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：仅依赖 TensorFlow 运行时环境
- 需要 mock/monkeypatch 的部分：无需外部依赖 mock，需验证 dataset_ops 模块正常

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 默认参数创建 int64 计数数据集
  2. 指定 start 和 step 参数验证序列生成
  3. 不同 dtype 参数（int32, float32, float64）验证
  4. 负 step 递减计数验证
  5. 浮点 start/step 参数验证

- 可选路径（中/低优先级合并为一组列表）：
  - 极端数值边界测试（极大/极小 start/step）
  - step=0 的特殊情况
  - 与 take() 操作组合使用验证
  - 多次调用创建独立数据集验证
  - 数据类型转换验证

- 已知风险/缺失信息（仅列条目，不展开）：
  - 未明确 start/step 参数类型约束范围
  - 未说明 dtype 参数支持的具体数据类型列表
  - 无限数据集内存消耗风险
  - CounterV1 和 Counter 函数的兼容性差异