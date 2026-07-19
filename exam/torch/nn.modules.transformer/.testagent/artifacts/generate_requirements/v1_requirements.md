# torch.nn.modules.transformer 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证 Transformer 架构（编码器、解码器）及其层组件的正确实现，包括注意力机制、前馈网络、层归一化、dropout 等核心功能
- 不在范围内的内容：底层 MultiheadAttention 实现细节、优化器训练逻辑、分布式训练、模型保存/加载

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - src/tgt: Tensor [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model] (batch_first=True)
  - src_mask/tgt_mask/memory_mask: ByteTensor/BoolTensor/FloatTensor [seq_len, seq_len] 或 [n*heads, seq_len, seq_len]
  - src_key_padding_mask/tgt_key_padding_mask/memory_key_padding_mask: ByteTensor/BoolTensor [batch_size, seq_len]
  - batch_first: bool (默认 False)
  - nhead: int (必须能被 d_model 整除)
  - num_encoder_layers/num_decoder_layers: int (默认 6)
  - dim_feedforward: int (默认 2048)
  - dropout: float (默认 0.1)
  - activation: str ("relu"/"gelu") 或 callable
  - layer_norm_eps: float (默认 1e-5)
  - norm_first: bool (默认 False)
  - enable_nested_tensor: bool (默认 True)

- 有效取值范围/维度/设备要求：
  - d_model 必须能被 nhead 整除
  - seq_len > 0, batch_size > 0
  - dropout ∈ [0, 1)
  - 支持 CPU/GPU 设备
  - 支持 float32/float64 数据类型
  - 嵌套张量仅当 enable_nested_tensor=True 且特定条件满足时启用

- 必需与可选组合：
  - src 必须提供，tgt 可选（仅编码器模式）
  - 掩码参数均为可选
  - 自定义编码器/解码器与标准实现互斥

- 随机性/全局状态要求：
  - dropout 引入随机性，需设置随机种子
  - 参数初始化使用 xavier_uniform_
  - 训练/推理模式影响 dropout 和 batch norm

## 3. 输出与判定
- 期望返回结构及关键字段：
  - Transformer: Tensor [tgt_seq_len, batch_size, d_model] 或 [batch_size, tgt_seq_len, d_model]
  - TransformerEncoder: Tensor [src_seq_len, batch_size, d_model]
  - TransformerDecoder: Tensor [tgt_seq_len, batch_size, d_model]
  - 各层输出形状与输入一致

- 容差/误差界（如浮点）：
  - 浮点误差容差：相对误差 1e-5，绝对误差 1e-7
  - 梯度计算数值稳定性验证
  - 不同设备（CPU/GPU）结果一致性

- 状态变化或副作用检查点：
  - 训练/推理模式切换不影响前向传播（除 dropout）
  - 参数更新后梯度清零
  - 嵌套张量优化路径正确触发

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - d_model 不能被 nhead 整除 → ValueError
  - 无效激活函数字符串 → ValueError
  - 张量维度不匹配 → RuntimeError
  - 无效掩码类型/形状 → RuntimeError
  - 嵌套张量条件不满足时警告

- 边界值（空、None、0 长度、极端形状/数值）：
  - seq_len=1 或 batch_size=1 的边界情况
  - 极大序列长度（内存边界）
  - dropout=0.0 或接近 1.0
  - 极小/极大数值输入（inf, nan 检测）
  - 全零掩码/全一掩码

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - PyTorch 库依赖
  - CUDA 设备（可选）
  - 足够内存处理大张量

- 需要 mock/monkeypatch 的部分：
  - 随机数生成器（控制 dropout）
  - 设备可用性检测
  - 嵌套张量优化条件判断
  - 自定义激活函数

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 标准 Transformer 前向传播（编码器+解码器）
  2. 仅编码器模式（tgt=None）
  3. 不同掩码类型（Byte/Bool/FloatTensor）正确应用
  4. batch_first=True/False 维度一致性
  5. 梯度计算和反向传播正确性

- 可选路径（中/低优先级合并为一组列表）：
  - 嵌套张量优化路径触发条件
  - 自定义编码器/解码器接口
  - 不同激活函数（relu/gelu/callable）
  - norm_first=True 配置
  - 极端形状（极小/极大序列长度）
  - 混合精度训练
  - 设备间迁移（CPU↔GPU）
  - 训练/推理模式切换

- 已知风险/缺失信息（仅列条目，不展开）：
  - 快速路径优化条件复杂性
  - 嵌套张量内存布局细节
  - 自定义组件边界条件
  - 多设备同步问题
  - 极端数值稳定性