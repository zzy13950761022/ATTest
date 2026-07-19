# torch.nn.modules.rnn 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证 RNN/LSTM/GRU 及其单元版本的正向传播、参数初始化、形状变换、序列处理功能
- 不在范围内的内容：反向传播梯度计算、优化器集成、训练收敛性、自定义激活函数

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - mode: str ('LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU')
  - input_size: int (>0)
  - hidden_size: int (>0)
  - num_layers: int (≥1, 默认1)
  - bias: bool (默认True)
  - batch_first: bool (默认False)
  - dropout: float (0-1, 默认0.0)
  - bidirectional: bool (默认False)
  - proj_size: int (≥0, 默认0, 仅LSTM支持)

- 有效取值范围/维度/设备要求：
  - dropout 仅在 num_layers > 1 时有效
  - proj_size 必须 < hidden_size (仅LSTM)
  - 输入形状：2D (seq_len, batch, input_size) 或 3D (batch, seq_len, input_size) 当 batch_first=True
  - 隐藏状态形状：多层时 (num_layers * num_directions, batch, hidden_size)
  - 支持 CPU/CUDA 设备

- 必需与可选组合：
  - 必需：input_size, hidden_size
  - 可选：h_0/c_0 初始状态（默认全零）
  - 组合约束：bidirectional 时输出维度为 2 * hidden_size

- 随机性/全局状态要求：
  - dropout 引入随机性
  - 权重初始化：均匀分布 U(-√k, √k), k = 1/hidden_size
  - 需要设置随机种子保证可重复性

## 3. 输出与判定
- 期望返回结构及关键字段：
  - RNN/GRU: (output, h_n) 元组
  - LSTM: (output, (h_n, c_n)) 元组
  - Cell 版本：单个隐藏状态或 (h, c) 元组
  - output 形状：(seq_len, batch, num_directions * hidden_size) 或 batch_first 格式

- 容差/误差界（如浮点）：
  - 浮点比较容差：1e-5 (单精度), 1e-7 (双精度)
  - 随机 dropout 允许统计性差异
  - 设备间（CPU/CUDA）结果一致性容差

- 状态变化或副作用检查点：
  - 权重参数初始化正确性
  - dropout 掩码生成一致性
  - 无外部 I/O 操作
  - 内存占用在预期范围内

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 输入张量维度错误（非2D/3D）
  - 输入/隐藏状态形状不匹配
  - proj_size 用于非 LSTM 类型
  - dropout 在 num_layers=1 时无效（警告）
  - 无效 mode 字符串
  - 负值或零值参数

- 边界值（空、None、0 长度、极端形状/数值）：
  - batch_size=0 或 seq_len=0
  - 极大 hidden_size (内存边界)
  - dropout=0.0 和 dropout=1.0 边界
  - proj_size=hidden_size (应失败)
  - 单层双向 RNN 输出形状验证

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - PyTorch C++ 后端 (_VF)
  - CUDA/cuDNN 库（可选）
  - 足够 GPU 内存（CUDA 测试）

- 需要 mock/monkeypatch 的部分：
  - 随机数生成器（dropout 测试）
  - CUDA 可用性检测
  - _VF 模块调用（隔离测试）
  - 内存分配失败场景

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基础 RNN/LSTM/GRU 正向传播形状正确性
  2. batch_first 格式输入输出转换
  3. 双向 RNN 输出维度验证
  4. dropout 在多层网络中的随机性
  5. LSTM 投影功能 (proj_size) 约束检查

- 可选路径（中/低优先级合并为一组列表）：
  - PackedSequence 输入处理
  - 不同设备（CPU/CUDA）结果一致性
  - 极端参数值（极大/极小 hidden_size）
  - 权重初始化分布验证
  - 单元版本 (RNNCell/LSTMCell/GRUCell) 独立测试
  - 梯度计算正确性（如包含反向传播）

- 已知风险/缺失信息（仅列条目，不展开）：
  - 多实体情况（8个主要类）
  - 类型注解不完整
  - CUDA/cuDNN 特定优化路径
  - 内存碎片化影响
  - 并行计算线程安全性