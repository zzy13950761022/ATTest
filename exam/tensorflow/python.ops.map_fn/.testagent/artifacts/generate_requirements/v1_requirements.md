# tensorflow.python.ops.map_fn 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 沿轴0展开elems张量，对每个元素应用fn函数
  - 将转换结果重新堆叠成结果张量
  - 支持单张量、嵌套结构、RaggedTensor和SparseTensor输入
  - 保持输入输出形状关系：[elems.shape[0]] + fn(elems[0]).shape
- 不在范围内的内容
  - 不测试map_fn_v2函数（包装器）
  - 不测试自动微分性能（仅验证back_prop参数功能）
  - 不测试实际GPU-CPU内存交换性能（仅验证swap_memory参数功能）

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - fn: callable，接受一个参数，参数结构与elems相同，无默认值
  - elems: tensor/sequence，至少包含一个张量，无默认值
  - dtype: Deprecated，已弃用，默认None
  - parallel_iterations: int/None，图构建默认10，eager默认1
  - back_prop: bool，默认True
  - swap_memory: bool，默认False
  - infer_shape: bool，默认True
  - name: str/None，默认None
  - fn_output_signature: tf.DType/TensorSpec/RaggedTensorSpec/SparseTensorSpec，默认None
- 有效取值范围/维度/设备要求
  - elems必须至少包含一个张量
  - 嵌套张量必须具有相同的外维度大小
  - RaggedTensor输入时，fn接收每行数据
  - SparseTensor输入时，fn接收每行数据（维度减1）
- 必需与可选组合
  - fn和elems为必需参数
  - 当fn输入输出签名不同时，必须指定fn_output_signature
  - 其他参数均为可选
- 随机性/全局状态要求
  - 无随机性要求
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - 张量或张量序列，堆叠fn应用于elems第一维展开的结果
  - 可能包含RaggedTensor和SparseTensor
  - 形状必须符合：[elems.shape[0]] + fn(elems[0]).shape
- 容差/误差界（如浮点）
  - 浮点运算使用默认TensorFlow容差
  - 形状检查使用严格相等
- 状态变化或副作用检查点
  - 验证back_prop=True时梯度可计算
  - 验证swap_memory参数不影响结果正确性
  - 验证infer_shape=True时形状一致性检查

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - elems为空序列时抛出ValueError
  - 嵌套张量外维度大小不一致时抛出ValueError
  - fn输入输出签名不同且未指定fn_output_signature时抛出TypeError
  - 使用已弃用的dtype参数时发出弃用警告
  - parallel_iterations<=0时抛出ValueError
- 边界值（空、None、0长度、极端形状/数值）
  - 零维张量输入（标量）
  - 空张量（shape包含0）
  - 大维度张量（内存边界）
  - 极端数值（inf, nan, 极大/极小值）
  - RaggedTensor不同ragged_rank
  - SparseTensor不同稀疏度

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow运行时环境
  - 支持GPU设备（可选）
  - 无网络/文件依赖
- 需要mock/monkeypatch的部分
  - `tensorflow.autograph`：用于自动图转换
  - `tensorflow.control_flow_ops.while_loop`：循环实现
  - `tensorflow.nest`：嵌套结构处理
  - `tensorflow.RaggedTensor`和`tensorflow.SparseTensor`特殊处理逻辑

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. 基本单张量映射功能验证
  2. 嵌套张量输入输出正确性
  3. fn输入输出签名不同时必须指定fn_output_signature
  4. RaggedTensor和SparseTensor特殊处理
  5. back_prop参数对梯度计算的影响
- 可选路径（中/低优先级合并为一组列表）
  - parallel_iterations参数在不同执行模式下的行为
  - swap_memory参数功能验证
  - infer_shape=False时形状推断行为
  - 大尺寸张量性能边界测试
  - 不同dtype转换组合测试
  - 零维和空张量边界情况
  - 极端数值处理
  - 名称前缀功能验证
- 已知风险/缺失信息（仅列条目，不展开）
  - eager模式下parallel_iterations>1不会真正并行执行
  - dtype参数已弃用但仍有代码路径
  - 模块包含map_fn和map_fn_v2两个函数
  - 部分参数类型注解缺失
  - 性能相比向量化操作较低