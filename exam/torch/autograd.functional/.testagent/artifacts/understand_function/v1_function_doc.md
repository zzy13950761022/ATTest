# torch.autograd.functional - 函数说明

## 1. 基本信息
- **FQN**: torch.autograd.functional
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/autograd/functional.py`
- **签名**: 模块（包含多个函数）
- **对象类型**: Python 模块

## 2. 功能概述
`torch.autograd.functional` 提供高阶自动微分函数，用于计算向量-Jacobian积、Jacobian矩阵、Hessian矩阵等。支持正向和反向模式自动微分，适用于需要高阶导数的科学计算场景。

## 3. 参数说明
模块包含以下核心函数：
- `vjp(func, inputs, v=None, create_graph=False, strict=False)`: 向量-Jacobian积
- `jvp(func, inputs, v=None, create_graph=False, strict=False)`: Jacobian-向量积
- `jacobian(func, inputs, create_graph=False, strict=False, vectorize=False, strategy="reverse-mode")`: Jacobian矩阵
- `hessian(func, inputs, create_graph=False, strict=False, vectorize=False, outer_jacobian_strategy="reverse-mode")`: Hessian矩阵
- `vhp(func, inputs, v=None, create_graph=False, strict=False)`: 向量-Hessian积
- `hvp(func, inputs, v=None, create_graph=False, strict=False)`: Hessian-向量积

## 4. 返回值
各函数返回元组，包含：
- 函数输出值
- 微分计算结果（形状与输入/输出匹配）

## 5. 文档要点
- 输入必须是 Tensor 或 Tensor 元组
- 支持 `create_graph` 参数控制是否创建计算图
- `strict` 模式检测输入输出独立性
- 支持向量化计算（实验性功能）
- 支持正向和反向模式策略

## 6. 源码摘要
- 使用 `_as_tuple` 统一处理输入输出
- 通过 `_grad_preprocess` 预处理梯度需求
- 调用 `torch.autograd.grad` 进行核心微分计算
- 使用 `_fill_in_zeros` 处理独立输入情况
- 依赖 `torch._vmap_internals._vmap` 实现向量化

## 7. 示例与用法
```python
def exp_reducer(x):
    return x.exp().sum(dim=1)

inputs = torch.rand(4, 4)
v = torch.ones(4)
vjp(exp_reducer, inputs, v)  # 计算向量-Jacobian积
jacobian(exp_reducer, inputs)  # 计算 Jacobian 矩阵
```

## 8. 风险与空白
- 目标为模块而非单个函数，包含多个微分函数实体
- `vectorize` 参数标记为实验性功能
- 正向模式 AD 要求 `vectorize=True`
- `strict=True` 与 `vectorize=True` 不兼容
- `create_graph=True` 与正向模式策略不兼容
- 缺少各函数详细的性能特征说明
- 未提供复杂张量形状的边界情况处理指南