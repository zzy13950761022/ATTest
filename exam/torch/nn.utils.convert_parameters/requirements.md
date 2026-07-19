# torch.nn.utils.convert_parameters 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - `parameters_to_vector`: 将参数张量迭代器展平连接为单个向量，保持设备一致性
  - `vector_to_parameters`: 将向量按原参数形状分割并赋值回参数，保持设备一致性
  - 双向转换应保持参数数值精度和形状不变
- 不在范围内的内容
  - 跨设备参数转换（不同GPU或CPU/GPU混合）
  - 非张量参数处理
  - 梯度传播和autograd行为
  - 性能基准测试（内存/时间消耗）

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `parameters_to_vector.parameters`: Iterable[torch.Tensor]，无默认值
  - `vector_to_parameters.vec`: torch.Tensor，无默认值
  - `vector_to_parameters.parameters`: Iterable[torch.Tensor]，无默认值
- 有效取值范围/维度/设备要求
  - 所有参数必须位于相同设备（CPU或同一GPU）
  - 参数张量可为任意形状，但必须可展平
  - vec长度必须等于参数总元素数
- 必需与可选组合
  - 所有参数均为必需，无可选参数
- 随机性/全局状态要求
  - 无随机性要求
  - 不依赖全局状态

## 3. 输出与判定
- 期望返回结构及关键字段
  - `parameters_to_vector`: 返回1D torch.Tensor，长度等于参数总元素数
  - `vector_to_parameters`: 无返回值，直接修改输入参数的data属性
- 容差/误差界（如浮点）
  - 数值转换应保持浮点精度（相对误差<1e-7）
  - 形状恢复应完全匹配
- 状态变化或副作用检查点
  - `vector_to_parameters`修改参数data属性
  - 参数梯度状态不受影响
  - 设备信息保持不变

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 非张量参数输入触发TypeError
  - 跨设备参数触发TypeError（通过`_check_param_device`）
  - vec长度与参数总元素数不匹配触发RuntimeError
  - 空迭代器输入触发ValueError
- 边界值（空、None、0长度、极端形状/数值）
  - 空迭代器（[]）
  - 零元素参数（torch.tensor([])）
  - 极端形状（超大维度、零维度）
  - 数值边界（inf、nan、极值）
  - None输入

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - CUDA设备（可选，用于GPU测试）
  - 无网络/文件依赖
- 需要mock/monkeypatch的部分
  - `torch.cat`: 用于验证连接逻辑
  - `torch.Tensor.view`: 用于验证展平操作
  - `torch.Tensor.get_device`: 用于设备检查
  - `torch.nn.utils.convert_parameters._check_param_device`: 核心验证逻辑

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. CPU设备参数正常双向转换
  2. GPU设备参数正常双向转换
  3. 混合形状参数转换验证
  4. 设备不一致异常触发
  5. vec长度不匹配异常触发
- 可选路径（中/低优先级合并为一组列表）
  - 空迭代器处理
  - 零元素参数处理
  - 极端形状参数（超大张量）
  - 数值边界测试（inf/nan）
  - 梯度传播影响（可选）
  - 性能压力测试（可选）
- 已知风险/缺失信息（仅列条目，不展开）
  - 缺少梯度传播行为文档
  - 未定义非连续内存张量处理
  - 缺少稀疏张量支持说明
  - 未明确复数张量支持
  - 缺少内存使用约束