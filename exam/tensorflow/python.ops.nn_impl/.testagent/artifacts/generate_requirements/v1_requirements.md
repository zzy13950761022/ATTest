# tensorflow.python.ops.nn_impl 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证神经网络函数模块中约30个核心函数的正确性，包括归一化、损失函数、激活函数等
- 不在范围内的内容：TensorFlow框架底层实现、分布式训练、自定义梯度、GPU/TPU特定优化

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - `l2_normalize`: x(Tensor), axis(int/tuple, default=-1), epsilon(float, default=1e-12)
  - `swish`: features(Tensor), beta(float, default=1.0)
  - `batch_normalization`: x(Tensor), mean(Tensor), variance(Tensor), offset(Tensor), scale(Tensor), variance_epsilon(float)
  - `log_poisson_loss`: targets(Tensor), log_input(Tensor), compute_full_loss(bool, default=False)
  - `moments`: x(Tensor), axes(int/tuple), shift(int, default=None), keepdims(bool, default=False)
- 有效取值范围/维度/设备要求：
  - 张量维度：1D-5D，支持广播
  - epsilon值：正浮点数，默认1e-12
  - axis参数：支持负数索引，-rank <= axis < rank
  - 数值范围：支持浮点数、整数、复数
- 必需与可选组合：
  - `batch_normalization`必需参数：x, mean, variance
  - `l2_normalize`可选参数：axis, epsilon
  - `swish`可选参数：beta
- 随机性/全局状态要求：
  - 无随机性依赖
  - 无全局状态修改

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 所有函数返回Tensor或Tensor列表
  - 保持输入张量形状（除指定axis外）
  - 数据类型与输入一致或自动提升
- 容差/误差界（如浮点）：
  - 浮点比较容差：1e-6（单精度），1e-12（双精度）
  - 归一化函数epsilon容差：1e-12
  - 数值稳定性边界：避免除零、溢出
- 状态变化或副作用检查点：
  - 无文件I/O操作
  - 无全局变量修改
  - 无设备状态改变

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非张量输入触发TypeError
  - 无效axis值触发ValueError
  - 形状不匹配触发ValueError
  - 不支持的数据类型触发TypeError
- 边界值（空、None、0长度、极端形状/数值）：
  - 空张量输入
  - 零值epsilon参数
  - 负值beta参数
  - 极大/极小数值（inf, -inf, nan）
  - 零长度维度
  - 单元素张量

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow核心库
  - 无网络/文件系统依赖
  - 无外部API调用
- 需要mock/monkeypatch的部分：
  - `tensorflow.python.ops.math_ops`（数学运算）
  - `tensorflow.python.ops.array_ops`（数组操作）
  - `tensorflow.python.ops.check_ops`（参数检查）
  - `tensorflow.python.ops.nn_ops`（神经网络操作）
  - `tensorflow.python.framework.ops.convert_to_tensor`（张量转换）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. `l2_normalize`基础归一化功能验证
  2. `swish`激活函数数值正确性
  3. `batch_normalization`训练/推理模式
  4. `moments`统计量计算准确性
  5. 损失函数梯度正确性检查
- 可选路径（中/低优先级合并为一组列表）：
  - 复数张量支持测试
  - 高维张量（5D+）处理
  - 混合精度计算验证
  - 弃用参数兼容性
  - 极端形状（超大/超小）处理
  - 广播机制边界情况
- 已知风险/缺失信息（仅列条目，不展开）：
  - 分布式计算支持不明确
  - 部分函数版本控制（v2后缀）
  - 浮点精度累积误差
  - 内存使用峰值未定义
  - 并发调用安全性未说明