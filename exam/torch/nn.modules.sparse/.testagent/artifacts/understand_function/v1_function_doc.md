# torch.nn.modules.sparse - 函数说明

## 1. 基本信息
- **FQN**: torch.nn.modules.sparse
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/nn/modules/sparse.py`
- **签名**: 模块包含两个类：Embedding 和 EmbeddingBag
- **对象类型**: Python 模块

## 2. 功能概述
- `Embedding`: 词嵌入查找表，存储固定字典大小的嵌入向量
- `EmbeddingBag`: 高效计算嵌入袋的求和/均值/最大值，无需实例化中间嵌入

## 3. 参数说明
**Embedding 类参数:**
- num_embeddings (int): 嵌入字典大小
- embedding_dim (int): 每个嵌入向量的维度
- padding_idx (int/None): 填充索引，不参与梯度更新
- max_norm (float/None): 最大范数约束
- norm_type (float): 范数类型，默认 2
- scale_grad_by_freq (bool): 按词频缩放梯度
- sparse (bool): 是否使用稀疏梯度

**EmbeddingBag 类参数:**
- 继承 Embedding 参数
- mode (str): "sum"/"mean"/"max"，聚合方式
- include_last_offset (bool): 偏移量格式

## 4. 返回值
- `Embedding.forward()`: 形状为 `(*, embedding_dim)` 的张量
- `EmbeddingBag.forward()`: 形状为 `(B, embedding_dim)` 的张量

## 5. 文档要点
- 输入必须是 IntTensor 或 LongTensor
- padding_idx 索引的嵌入向量初始化为零
- max_norm 不为 None 时会原地修改权重张量
- 稀疏梯度仅支持特定优化器

## 6. 源码摘要
- Embedding.forward() 调用 F.embedding
- EmbeddingBag.forward() 调用 F.embedding_bag
- 初始化时验证 padding_idx 范围
- 权重使用正态分布初始化
- 包含 from_pretrained 类方法

## 7. 示例与用法
- Embedding: 10个嵌入向量，每个维度3
- EmbeddingBag: 支持 offsets 和 per_sample_weights
- 包含 padding_idx 使用示例

## 8. 风险与空白
- 模块包含两个主要类，测试需覆盖两者
- 未明确指定输入张量的设备要求
- 稀疏梯度支持的优化器有限
- max_norm 原地修改权重的副作用
- 边界情况：空袋、负索引、越界索引
- 需要测试不同 mode 参数的行为
- 需要验证 per_sample_weights 与 mode 的兼容性