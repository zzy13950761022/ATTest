# torch.nn.utils.rnn - 函数说明

## 1. 基本信息
- **FQN**: torch.nn.utils.rnn
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/nn/utils/rnn.py`
- **签名**: 模块（包含多个函数和类）
- **对象类型**: module

## 2. 功能概述
PyTorch RNN工具模块，提供处理变长序列的函数。核心功能包括打包/解包填充序列、处理PackedSequence对象。用于RNN输入的高效批处理。

## 3. 参数说明
模块包含多个函数，主要参数：
- `pack_padded_sequence`: input(Tensor), lengths(Tensor/list), batch_first(bool), enforce_sorted(bool)
- `pad_packed_sequence`: sequence(PackedSequence), batch_first(bool), padding_value(float), total_length(int)
- `pad_sequence`: sequences(List[Tensor]), batch_first(bool), padding_value(float)
- `unpad_sequence`: padded_sequences(Tensor), lengths(Tensor), batch_first(bool)
- `pack_sequence`: sequences(List[Tensor]), enforce_sorted(bool)
- `unpack_sequence`: packed_sequences(PackedSequence)

## 4. 返回值
- `pack_padded_sequence`: PackedSequence对象
- `pad_packed_sequence`: (Tensor, Tensor)元组（填充序列和长度）
- `pad_sequence`: 填充后的Tensor
- `unpad_sequence`: List[Tensor]
- `pack_sequence`: PackedSequence对象
- `unpack_sequence`: List[Tensor]

## 5. 文档要点
- PackedSequence.data可在任意设备和dtype上
- sorted_indices/unsorted_indices必须是torch.int64，与data同设备
- batch_sizes必须是CPU上的torch.int64张量
- enforce_sorted=True时序列需按长度降序排列
- total_length必须≥最长序列长度

## 6. 源码摘要
- 依赖_VF模块的C++实现（_pack_padded_sequence, _pad_packed_sequence）
- PackedSequence继承自namedtuple，添加设备转换方法
- pack_padded_sequence调用_VF._pack_padded_sequence
- pad_packed_sequence调用_VF._pad_packed_sequence
- 无I/O操作，无随机性，无全局状态修改

## 7. 示例与用法
```python
# pack_padded_sequence示例
seq = torch.tensor([[1,2,0], [3,0,0], [4,5,6]])
lens = [2, 1, 3]
packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)

# pad_packed_sequence示例
seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
```

## 8. 风险与空白
- 模块包含8个主要函数，需分别测试
- 部分函数依赖C++实现，需验证边界条件
- 设备转换逻辑复杂（CPU/GPU，dtype转换）
- 需要测试enforce_sorted=False时的排序行为
- 需要验证total_length参数边界情况
- 需要测试不同dtype和设备的兼容性
- 缺少详细的异常类型文档
- 需要测试空序列和零长度序列