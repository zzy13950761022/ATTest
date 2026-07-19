# torch.nn.modules.distance - 函数说明

## 1. 基本信息
- **FQN**: torch.nn.modules.distance
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/nn/modules/distance.py`
- **签名**: 模块（包含两个类）
- **对象类型**: module

## 2. 功能概述
该模块提供两个距离计算类：PairwiseDistance计算向量/矩阵间的p-范数距离，CosineSimilarity计算指定维度上的余弦相似度。两者都是torch.nn.Module的子类，封装了对应的functional函数。

## 3. 参数说明
**PairwiseDistance类：**
- p (float/2.0): 范数度，可为负值
- eps (float/1e-6): 避免除零的小值
- keepdim (bool/False): 是否保持向量维度

**CosineSimilarity类：**
- dim (int/1): 计算余弦相似度的维度
- eps (float/1e-8): 避免除零的小值

## 4. 返回值
- 两个类的forward方法都返回Tensor
- PairwiseDistance输出形状：(N)或()，keepdim=True时为(N,1)或(1)
- CosineSimilarity输出形状：(*1, *2)，去除dim维度

## 5. 文档要点
- PairwiseDistance使用p-范数公式：dist(x,y)=‖x-y+εe‖ₚ
- CosineSimilarity公式：x₁·x₂/max(‖x₁‖₂·‖x₂‖₂, ε)
- 张量形状约束：输入必须匹配dim维度大小
- 支持广播机制

## 6. 源码摘要
- 两个类都继承自torch.nn.Module
- forward方法直接调用torch.nn.functional对应函数
- PairwiseDistance调用F.pairwise_distance
- CosineSimilarity调用F.cosine_similarity
- 无I/O、随机性或全局状态副作用

## 7. 示例与用法（如有）
**PairwiseDistance示例：**
```python
pdist = nn.PairwiseDistance(p=2)
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
output = pdist(input1, input2)
```

**CosineSimilarity示例：**
```python
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
output = cos(input1, input2)
```

## 8. 风险与空白
- 模块包含两个独立类，需要分别测试
- 未明确p为负值时的具体行为
- 缺少对极端值（如inf/nan）的处理说明
- 未说明设备（CPU/GPU）兼容性
- 广播行为的边界条件未详细说明
- 缺少对输入dtype限制的明确文档