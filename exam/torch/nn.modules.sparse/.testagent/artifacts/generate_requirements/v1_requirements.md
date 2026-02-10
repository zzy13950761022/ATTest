# torch.nn.modules.sparse 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - Embedding: 词嵌入查找表，支持索引到向量映射
  - EmbeddingBag: 高效聚合多个嵌入向量（sum/mean/max）
  - 支持稀疏梯度、范数约束、填充索引等特性
- 不在范围内的内容
  - 其他稀疏模块（如稀疏卷积）
  - 底层 F.embedding/F.embedding_bag 实现细节
  - 特定优化器的稀疏梯度支持

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - num_embeddings: int, 必需, 嵌入字典大小
  - embedding_dim: int, 必需, 嵌入向量维度
  - padding_idx: int/None, 可选, 默认None
  - max_norm: float/None, 可选, 默认None
  - norm_type: float, 可选, 默认2.0
  - scale_grad_by_freq: bool, 可选, 默认False
  - sparse: bool, 可选, 默认False
  - mode: str, EmbeddingBag特有, "sum"/"mean"/"max", 默认"mean"
  - include_last_offset: bool, EmbeddingBag特有, 默认False
- 有效取值范围/维度/设备要求
  - num_embeddings > 0
  - embedding_dim > 0
  - padding_idx ∈ [-num_embeddings, num_embeddings-1] 或 None
  - 输入张量必须是 IntTensor 或 LongTensor
  - 支持CPU和CUDA设备
- 必需与可选组合
  - num_embeddings 和 embedding_dim 必需同时提供
  - per_sample_weights 仅与特定 mode 兼容
- 随机性/全局状态要求
  - 权重使用正态分布初始化
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - Embedding.forward(): 形状 (*, embedding_dim) 的张量
  - EmbeddingBag.forward(): 形状 (B, embedding_dim) 的张量
  - B为批次大小，*为输入形状
- 容差/误差界（如浮点）
  - 浮点比较容差: 1e-5
  - 聚合操作（sum/mean/max）需验证数值正确性
- 状态变化或副作用检查点
  - max_norm 不为 None 时会原地修改权重张量
  - padding_idx 对应的嵌入向量应保持为零且无梯度
  - 稀疏梯度模式下权重更新行为

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - num_embeddings ≤ 0 触发 ValueError
  - embedding_dim ≤ 0 触发 ValueError
  - padding_idx 超出范围触发 ValueError
  - 输入张量非 IntTensor/LongTensor 触发 TypeError
  - 索引越界触发 RuntimeError
  - 不兼容的 per_sample_weights 和 mode 组合
- 边界值（空、None、0 长度、极端形状/数值）
  - 空输入张量（零元素）
  - 极端形状：大 num_embeddings/embedding_dim
  - 边界索引：0, num_embeddings-1, -1
  - max_norm 为 0 或极小值
  - 包含 padding_idx 的输入序列

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - PyTorch 库依赖
  - CUDA 设备（可选）
  - 无网络/文件依赖
- 需要 mock/monkeypatch 的部分
  - F.embedding/F.embedding_bag 调用验证
  - 正态分布初始化
  - 稀疏梯度优化器行为

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. Embedding 基础正向传播与形状验证
  2. EmbeddingBag 三种聚合模式功能正确性
  3. padding_idx 特殊处理与梯度隔离
  4. max_norm 范数约束与权重修改
  5. 稀疏梯度模式与密集梯度模式对比
- 可选路径（中/低优先级合并为一组列表）
  - 极端形状和大规模参数测试
  - 不同设备（CPU/CUDA）兼容性
  - per_sample_weights 与各种 mode 组合
  - include_last_offset 不同设置
  - scale_grad_by_freq 梯度缩放效果
  - 边界索引和越界处理
  - from_pretrained 类方法
- 已知风险/缺失信息（仅列条目，不展开）
  - 稀疏梯度仅支持特定优化器
  - max_norm 原地修改权重的副作用
  - 空袋（empty bag）处理逻辑
  - 负索引的语义解释
  - 不同 PyTorch 版本行为差异