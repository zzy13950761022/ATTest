# torch.nn.utils.rnn 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试RNN工具模块的序列打包/解包功能，验证PackedSequence对象创建、填充序列处理、设备/数据类型兼容性
- 不在范围内的内容：RNN模型训练、梯度计算、自定义RNN单元、分布式训练

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - pack_padded_sequence: input(Tensor[B×T×*]/[T×B×*]), lengths(Tensor/list[int]), batch_first(bool=False), enforce_sorted(bool=True)
  - pad_packed_sequence: sequence(PackedSequence), batch_first(bool=False), padding_value(float=0.0), total_length(int|None)
  - pad_sequence: sequences(List[Tensor]), batch_first(bool=False), padding_value(float=0.0)
  - unpad_sequence: padded_sequences(Tensor), lengths(Tensor), batch_first(bool=False)
  - pack_sequence: sequences(List[Tensor]), enforce_sorted(bool=True)
  - unpack_sequence: packed_sequences(PackedSequence)

- 有效取值范围/维度/设备要求：
  - PackedSequence.data支持任意设备和dtype
  - sorted_indices/unsorted_indices必须是torch.int64，与data同设备
  - batch_sizes必须是CPU上的torch.int64张量
  - enforce_sorted=True时序列需按长度降序排列
  - total_length必须≥最长序列长度

- 必需与可选组合：
  - pack_padded_sequence: input和lengths必需，batch_first和enforce_sorted可选
  - pad_packed_sequence: sequence必需，其他可选
  - 所有函数支持CPU和CUDA设备

- 随机性/全局状态要求：无随机性，无全局状态修改

## 3. 输出与判定
- 期望返回结构及关键字段：
  - pack_padded_sequence: PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)
  - pad_packed_sequence: (Tensor, Tensor)元组（填充序列和长度）
  - pad_sequence: 填充后的Tensor
  - unpad_sequence: List[Tensor]
  - pack_sequence: PackedSequence对象
  - unpack_sequence: List[Tensor]

- 容差/误差界（如浮点）：浮点误差在1e-6范围内，整数精确匹配

- 状态变化或副作用检查点：
  - 验证输入张量不被修改
  - 验证输出张量设备与输入一致
  - 验证PackedSequence属性完整性

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 输入非张量或张量列表
  - lengths长度与batch_size不匹配
  - enforce_sorted=True但序列未排序
  - total_length小于实际最大长度
  - 空序列或零长度序列
  - 设备不匹配（如lengths在GPU但input在CPU）

- 边界值（空、None、0 长度、极端形状/数值）：
  - 空列表输入
  - 零长度序列
  - 单元素序列
  - 极大batch_size（接近内存限制）
  - 极大序列长度
  - 负padding_value
  - 非整数lengths

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：无外部资源，支持CUDA设备（可选）

- 需要 mock/monkeypatch 的部分：
  - `torch._VF._pack_padded_sequence`（C++实现）
  - `torch._VF._pad_packed_sequence`（C++实现）
  - `torch.cuda.is_available()`（设备检测）
  - `torch.get_default_dtype()`（数据类型默认值）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. pack_padded_sequence基本功能与pad_packed_sequence逆操作
  2. enforce_sorted=True/False的排序行为验证
  3. 不同设备（CPU/CUDA）和dtype的兼容性
  4. batch_first=True/False的维度转换
  5. total_length参数边界和默认行为

- 可选路径（中/低优先级合并为一组列表）：
  - 极端形状和大小的性能测试
  - 混合精度（float16/float32/float64）转换
  - 嵌套序列处理
  - 与RNN模型集成测试
  - 内存泄漏和性能基准

- 已知风险/缺失信息（仅列条目，不展开）：
  - 部分函数依赖C++实现，边界条件验证不足
  - 缺少详细的异常类型文档
  - 设备转换逻辑复杂度高
  - 空序列和零长度序列处理未充分文档化
  - 多线程环境下的行为未定义