# tensorflow.python.ops.ragged.ragged_concat_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - `concat`: 沿指定维度连接不规则张量列表，保持输入张量秩不变
  - `stack`: 沿指定维度堆叠不规则张量列表，输出张量秩增加1
  - 支持混合规则张量和不规则张量输入
  - 支持任意形状输入（与标准 tf.concat/tf.stack 不同）
- 不在范围内的内容
  - 非张量类型输入转换
  - 动态 axis 值（axis 必须静态已知）
  - 跨设备/分布式张量操作

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `values`: List[RaggedOrDense]，不能为空列表
  - `axis`: int，必须静态已知，支持负值（需至少一个输入秩静态已知）
  - `name`: str/None，可选操作名称
  - `stack` 函数：axis 默认值为 0
- 有效取值范围/维度/设备要求
  - 所有输入必须具有相同秩和 dtype
  - axis 必须在 [-rank, rank) 范围内
  - 仅支持 CPU/GPU 张量，无特殊设备要求
- 必需与可选组合
  - `values` 为必需参数，不能为空
  - `axis` 为必需参数（stack 有默认值）
  - `name` 为可选参数
- 随机性/全局状态要求
  - 无随机性操作
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回 `RaggedTensor` 对象
  - `concat`: 输出秩与输入相同
  - `stack`: 输出秩为输入秩+1（R>0时），R==0 时返回 1D Tensor
- 容差/误差界（如浮点）
  - 数值精度遵循 TensorFlow 浮点运算标准
  - 无特殊容差要求
- 状态变化或副作用检查点
  - 无 I/O 操作
  - 无全局状态修改
  - 无内存泄漏

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 空 values 列表 → ValueError
  - 输入张量秩不同 → ValueError
  - 输入 dtype 不同 → ValueError
  - axis 超出有效范围 → ValueError
  - 负 axis 值且无输入秩静态已知 → ValueError
  - 非张量类型输入 → TypeError
- 边界值（空、None、0 长度、极端形状/数值）
  - 单元素 values 列表
  - 秩为0的标量张量
  - 秩为1的向量张量
  - 空内部列表的 RaggedTensor
  - 极大/极小数值
  - axis=0, axis=1, axis>1 三种情况

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 运行时环境
  - 无网络/文件系统依赖
- 需要 mock/monkeypatch 的部分
  - `tensorflow.python.ops.ragged.ragged_tensor.RaggedTensor`
  - `tensorflow.python.ops.array_ops.concat`
  - `tensorflow.python.ops.array_ops.stack`
  - `tensorflow.python.ops.check_ops.assert_positive`
  - `tensorflow.python.ops.math_ops.range`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. concat 基本功能：混合 Ragged/Dense 张量沿 axis=0/1 连接
  2. stack 基本功能：混合 Ragged/Dense 张量沿 axis=0/1 堆叠
  3. 边界处理：空 values 列表、单输入、秩0/1张量
  4. 错误路径：秩不匹配、dtype 不匹配、axis 越界
  5. 负 axis 值处理：有/无静态已知秩的情况
- 可选路径（中/低优先级合并为一组列表）
  - 极端形状测试：超大维度、深层嵌套
  - 混合数值类型：int/float/bool 组合
  - 性能基准：大规模张量连接/堆叠
  - 与标准 tf.concat/tf.stack 行为对比
  - 跨版本兼容性测试
- 已知风险/缺失信息（仅列条目，不展开）
  - `RaggedOrDense` 类型定义未明确
  - `row_splits_dtype` 匹配过程细节缺失
  - 内存使用模式未文档化
  - 梯度计算行为未覆盖