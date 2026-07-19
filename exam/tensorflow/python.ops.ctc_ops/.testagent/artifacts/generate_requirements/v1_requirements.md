# tensorflow.python.ops.ctc_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证CTC损失计算和束搜索解码的正确性，包括序列标注任务的负对数概率计算和最优路径解码
- 不在范围内的内容：端到端模型训练、自定义CTC算法实现、非标准稀疏张量格式

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - labels: SparseTensor[int32]，稀疏标签序列
  - inputs: Tensor[float]，3D对数概率张量，形状依赖time_major
  - sequence_length: vector[int32]，[batch_size]，序列长度
  - preprocess_collapse_repeated: bool，默认False，预处理合并重复标签
  - ctc_merge_repeated: bool，默认True，CTC计算合并重复标签
  - ignore_longer_outputs_than_inputs: bool，默认False
  - time_major: bool，默认True，输入张量时间主序
  - beam_width: int，默认100，束搜索宽度
  - top_paths: int，默认1，输出路径数
  - merge_repeated: bool，默认True，解码合并重复标签

- 有效取值范围/维度/设备要求：
  - inputs最内层维度：num_classes = num_labels + 1（空白标签）
  - 标签索引范围：[0, num_labels)
  - sequence_length(b) <= time for all b
  - 支持两种形状格式：[max_time, batch_size, num_classes]（time_major=True）或[batch_size, max_time, num_classes]
  - 支持CPU/GPU设备

- 必需与可选组合：
  - labels, inputs, sequence_length为必需参数
  - preprocess_collapse_repeated和ctc_merge_repeated组合需测试所有4种情况
  - beam_width和top_paths仅用于解码函数

- 随机性/全局状态要求：
  - 无随机性要求
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段：
  - ctc_loss: 1D float Tensor [batch_size]，负对数概率损失值
  - ctc_beam_search_decoder: tuple (decoded, log_probabilities)
    - decoded: SparseTensor列表，长度top_paths
    - log_probabilities: float矩阵 [batch_size x top_paths]

- 容差/误差界（如浮点）：
  - 浮点误差容差：1e-6（单精度）
  - 损失值非负验证
  - 解码路径概率单调递减验证（top_paths>1时）

- 状态变化或副作用检查点：
  - 无外部状态变化
  - 稀疏张量格式保持正确性
  - 输入张量不被修改

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - labels非SparseTensor类型
  - inputs维度不为3
  - sequence_length长度与batch_size不匹配
  - 标签值超出[0, num_labels)范围
  - sequence_length值大于max_time
  - num_classes <= num_labels（缺少空白标签维度）

- 边界值（空、None、0长度、极端形状/数值）：
  - batch_size=0（空批次）
  - sequence_length=0（零长度序列）
  - max_time=1（最小时间维度）
  - num_labels=1（最小标签集）
  - beam_width=1（最小束宽）
  - top_paths=0（无路径输出）
  - 极大batch_size和max_time组合
  - 浮点极值（inf, nan）输入

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow C++扩展：gen_ctc_ops
  - GPU CUDA库（GPU运行时）

- 需要mock/monkeypatch的部分：
  - `tensorflow.python.ops.gen_ctc_ops.ctc_loss_v2`
  - `tensorflow.python.ops.gen_ctc_ops.ctc_beam_search_decoder`
  - `tensorflow.python.framework.ops.device`（设备上下文）
  - `tensorflow.python.ops.sparse_ops.sparse_tensor_to_dense`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 基本CTC损失计算：标准形状和参数
  2. 束搜索解码：beam_width=100, top_paths=1
  3. 时间主序和批次主序格式兼容性
  4. preprocess_collapse_repeated和ctc_merge_repeated组合测试
  5. 边界条件：空批次和零长度序列

- 可选路径（中/低优先级合并为一组列表）：
  - 不同beam_width值（1, 10, 1000）
  - top_paths>1的多路径输出
  - merge_repeated=False的解码行为
  - ignore_longer_outputs_than_inputs=True场景
  - 不同设备（CPU/GPU）结果一致性
  - 梯度计算正确性验证
  - 稀疏张量格式异常处理

- 已知风险/缺失信息（仅列条目，不展开）：
  - preprocess_collapse_repeated=True, ctc_merge_repeated=True组合未测试
  - 文档中缺少端到端使用示例
  - 需要验证不同TensorFlow版本的兼容性
  - 需要测试大batch_size下的内存使用
  - 需要验证GPU加速的正确性