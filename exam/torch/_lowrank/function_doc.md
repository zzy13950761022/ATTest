# torch._lowrank - 函数说明

## 1. 基本信息
- **FQN**: torch._lowrank
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/_lowrank.py`
- **签名**: 模块包含多个函数
- **对象类型**: module

## 2. 功能概述
实现低秩矩阵的线性代数算法。提供近似奇异值分解和主成分分析功能。基于随机算法高效处理大规模低秩矩阵。

## 3. 参数说明
模块包含三个主要函数：

**get_approximate_basis(A, q, niter=2, M=None)**
- A (Tensor): 输入张量，形状 `(*, m, n)`
- q (int): 子空间维度，q ≤ min(m, n)
- niter (int, optional): 子空间迭代次数，非负整数，默认2
- M (Tensor, optional): 均值张量，形状 `(*, 1, n)`

**svd_lowrank(A, q=6, niter=2, M=None)**
- A (Tensor): 输入张量，形状 `(*, m, n)`
- q (int, optional): 略微高估的秩，默认6
- niter (int, optional): 子空间迭代次数，非负整数，默认2
- M (Tensor, optional): 均值张量，形状 `(*, 1, n)`

**pca_lowrank(A, q=None, center=True, niter=2)**
- A (Tensor): 输入张量，形状 `(*, m, n)`
- q (int, optional): 略微高估的秩，默认 min(6, m, n)
- center (bool, optional): 是否中心化，默认True
- niter (int, optional): 子空间迭代次数，非负整数，默认2

## 4. 返回值
- **get_approximate_basis**: 返回正交基张量 Q，形状 `(*, m, q)`
- **svd_lowrank**: 返回元组 (U, S, V)，其中 U 形状 `(*, m, q)`，S 形状 `(*, q)`，V 形状 `(*, n, q)`
- **pca_lowrank**: 返回元组 (U, S, V)，与 svd_lowrank 类似

## 5. 文档要点
- 输入矩阵假设为低秩矩阵
- 对于稠密矩阵，建议使用 torch.linalg.svd（性能高10倍）
- 低秩SVD适用于 torch.linalg.svd 无法处理的大规模稀疏矩阵
- 需要可重复结果时，重置伪随机数生成器种子
- q 的选择准则：k ≤ q ≤ min(2*k, m, n)，其中 k 是未知的真实秩

## 6. 源码摘要
- 基于 Halko et al, 2009 的算法 4.4 和 5.1
- 使用随机投影和子空间迭代
- 依赖 torch.linalg.qr 和 torch.linalg.svd
- 处理稀疏和稠密张量
- 包含 torch 函数分发机制（handle_torch_function）

## 7. 示例与用法（如有）
- 文档中提供数学公式和算法描述
- 包含 PCA 与 SVD 的关系说明
- 提供 q 参数选择指南

## 8. 风险与空白
- 模块包含多个函数，需要分别测试
- 随机性：结果依赖随机数生成器状态
- 边界条件：q 值范围验证（0 ≤ q ≤ min(m, n)）
- 稀疏矩阵处理：仅 pca_lowrank 支持稀疏输入
- 性能警告：对于稠密矩阵，低秩SVD比完整SVD慢10倍
- 缺少具体数值示例
- 需要测试不同 dtype（浮点类型）和 device（CPU/GPU）
- 需要验证中心化参数对结果的影响