# tensorflow.python.ops.ragged.ragged_factory_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - `constant`: 从嵌套Python列表构造常量RaggedTensor，支持dtype推断和形状控制
  - `constant_value`: 从嵌套Python列表构造RaggedTensorValue（NumPy数组）
  - `placeholder`: 创建RaggedTensor占位符，用于TensorFlow 1.x的feed_dict
- 不在范围内的内容
  - RaggedTensor的其他操作方法（如数学运算、切片等）
  - TensorFlow 2.x的eager模式占位符替代方案
  - 非Python列表输入（如Tensor、SparseTensor）的转换

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `constant`/`constant_value`:
    - `pylist`: list/tuple/np.ndarray，嵌套结构，标量元素
    - `dtype`: tf.DType/None，默认None（自动推断）
    - `ragged_rank`: int/None，默认None，必须非负且小于嵌套深度K
    - `inner_shape`: tuple/None，默认None（自动推断）
    - `name`: str/None，默认None
    - `row_splits_dtype`: tf.int32/tf.int64或"int32"/"int64"，默认int64
  - `placeholder`:
    - `dtype`: tf.DType，必需
    - `ragged_rank`: int，必需
    - `value_shape`: tuple/None，默认None
    - `name`: str/None，默认None

- 有效取值范围/维度/设备要求
  - 所有标量值必须具有相同的嵌套深度K
  - 空列表的最大深度决定K值
  - ragged_rank必须小于K
  - inner_shape必须与pylist内容兼容
  - 标量值必须与dtype兼容

- 必需与可选组合
  - `constant`/`constant_value`: pylist必需，其他可选
  - `placeholder`: dtype和ragged_rank必需，value_shape和name可选

- 随机性/全局状态要求
  - 无随机性要求
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - `constant`: tf.RaggedTensor，具有正确的ragged_rank和inner_shape
  - `constant_value`: tf.RaggedTensorValue或numpy.array，包含values和row_splits
  - `placeholder`: tf.RaggedTensor占位符，不能直接求值

- 容差/误差界（如浮点）
  - 数值精度遵循TensorFlow浮点运算标准
  - dtype转换遵循TensorFlow类型转换规则

- 状态变化或副作用检查点
  - 无副作用
  - 不修改输入数据
  - 不改变全局状态

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - pylist包含不一致的嵌套深度
  - ragged_rank大于等于嵌套深度K
  - inner_shape与pylist内容不兼容
  - dtype与标量值类型不兼容
  - 无效的row_splits_dtype值
  - placeholder在TensorFlow 2.x eager模式下使用

- 边界值（空、None、0长度、极端形状/数值）
  - 空列表：[]、[[]]、[[[]]]等
  - 单元素列表：[[1]]、[[[1]]]等
  - 极大嵌套深度（递归限制）
  - 极大列表长度（内存限制）
  - 极端数值：inf、nan、极大/极小值
  - 混合数值类型（int、float、complex）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow库
  - NumPy库（用于constant_value）
  - 无网络/文件/设备特定依赖

- 需要mock/monkeypatch的部分
  - `tensorflow.python.ops.ragged.ragged_tensor.RaggedTensor.from_row_splits`
  - `tensorflow.python.ops.constant_op.constant`
  - `tensorflow.python.framework.ops.Tensor`（用于placeholder测试）
  - `tensorflow.python.ops.ragged.ragged_tensor_value.RaggedTensorValue`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. constant基本功能：从简单嵌套列表创建RaggedTensor
  2. constant_value基本功能：创建RaggedTensorValue
  3. dtype自动推断：不同类型标量的正确类型推断
  4. ragged_rank验证：有效和无效ragged_rank参数处理
  5. 错误处理：不一致嵌套深度引发的异常

- 可选路径（中/低优先级合并为一组列表）
  - placeholder功能测试（仅限TensorFlow 1.x）
  - inner_shape参数的各种组合
  - row_splits_dtype参数测试（int32 vs int64）
  - 复杂嵌套结构（混合元组/列表/np.ndarray）
  - 边界情况：空列表、单元素、极大嵌套
  - 性能测试：大尺寸输入处理

- 已知风险/缺失信息（仅列条目，不展开）
  - placeholder在TensorFlow 2.x中的兼容性问题
  - 复杂混合类型输入的明确约束
  - 递归深度限制的具体数值
  - 内存使用优化策略
  - 与TensorFlow其他ragged操作的集成边界