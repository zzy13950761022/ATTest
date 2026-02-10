# tensorflow.python.keras.layers.rnn_cell_wrapper_v2 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证三个包装器类（DropoutWrapper、ResidualWrapper、DeviceWrapper）正确包装RNN单元
  - 确保包装器与TF v2 API兼容，继承自recurrent.AbstractRNNCell
  - 测试包装器序列化/反序列化功能（get_config/from_config）
  - 验证包装器通过tf.nn.* API正确导出
- 不在范围内的内容
  - Keras LSTM cell与DropoutWrapper的兼容性（已知不支持）
  - 底层RNN单元的内部实现细节
  - 非TF v2兼容的旧API

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - DropoutWrapper: cell(RNNCell), input_keep_prob(float,0-1,1.0), output_keep_prob(float,0-1,1.0), state_keep_prob(float,0-1,1.0), variational_recurrent(bool,False), input_size(int,None), dtype(tf.DType,None), seed(int,None), dropout_state_filter_visitor(callable,None)
  - ResidualWrapper: cell(RNNCell)
  - DeviceWrapper: cell(RNNCell), device(str)
- 有效取值范围/维度/设备要求
  - dropout概率必须在[0,1]区间内
  - device参数必须是有效设备字符串（如"/cpu:0", "/gpu:0"）
  - cell必须是有效的RNNCell实例
- 必需与可选组合
  - cell参数对所有包装器都是必需的
  - dropout概率参数可选，默认值为1.0（无dropout）
  - variational_recurrent、input_size、dtype、seed、dropout_state_filter_visitor为可选参数
- 随机性/全局状态要求
  - DropoutWrapper的seed参数控制随机性
  - 需要测试随机dropout的可重复性

## 3. 输出与判定
- 期望返回结构及关键字段
  - 包装器返回包装后的RNNCell实例
  - 实例必须正确响应call()方法，返回(output, next_state)
  - get_config()返回包含所有参数的字典
- 容差/误差界（如浮点）
  - dropout概率浮点比较容差：1e-6
  - 输出数值容差：1e-5（与未包装cell对比）
- 状态变化或副作用检查点
  - 包装器不应修改原始cell的内部状态
  - DeviceWrapper必须正确设置设备放置
  - ResidualWrapper必须正确添加残差连接

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - cell参数不是RNNCell类型：TypeError
  - dropout概率超出[0,1]范围：ValueError
  - 无效设备字符串：ValueError/InvalidArgumentError
  - 输入维度与cell不匹配：ValueError
- 边界值（空、None、0长度、极端形状/数值）
  - cell=None：TypeError
  - dropout概率=0.0或1.0：边界测试
  - 极端形状输入（如batch_size=1, time_steps=1000）
  - 高维输入测试（3D+形状）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow v2.x环境
  - 可能需要GPU设备测试DeviceWrapper
  - 依赖tensorflow.python.keras.layers.legacy_rnn.rnn_cell_wrapper_impl
- 需要mock/monkeypatch的部分
  - 随机数生成器（测试dropout可重复性）
  - 设备可用性检查
  - tf.nn.* API导出机制

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. 三个包装器类与基本RNNCell（如BasicRNNCell）的集成
  2. DropoutWrapper概率参数边界值（0.0, 0.5, 1.0）测试
  3. 包装器序列化/反序列化循环测试
  4. DeviceWrapper设备放置正确性验证
  5. ResidualWrapper维度匹配和残差计算正确性
- 可选路径（中/低优先级合并为一组列表）
  - 与不同RNN cell类型（LSTMCell, GRUCell）的兼容性
  - variational_recurrent=True模式测试
  - dropout_state_filter_visitor回调函数测试
  - 多批次、变长序列输入测试
  - 混合精度（float16/float32）支持测试
- 已知风险/缺失信息（仅列条目，不展开）
  - DropoutWrapper不支持keras LSTM cell
  - 模块未来可能被弃用
  - 参数类型约束文档不完整
  - 性能影响未量化