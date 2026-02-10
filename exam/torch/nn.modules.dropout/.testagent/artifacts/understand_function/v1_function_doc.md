# torch.nn.modules.dropout - 函数说明

## 1. 基本信息
- **FQN**: torch.nn.modules.dropout
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/nn/modules/dropout.py`
- **签名**: 模块包含多个类，无单一函数签名
- **对象类型**: Python 模块

## 2. 功能概述
该模块提供多种 Dropout 层实现，用于神经网络正则化。包含标准 Dropout、空间 Dropout（1D/2D/3D）和 AlphaDropout 变体。所有类继承自 `_DropoutNd` 基类，在训练时随机屏蔽输入元素，评估时执行恒等映射。

## 3. 参数说明
**基类 _DropoutNd:**
- p (float, 默认 0.5): dropout 概率，范围 [0, 1]
- inplace (bool, 默认 False): 是否原地操作

**各子类特定约束:**
- Dropout: 任意形状输入
- Dropout1d: 输入形状 (N, C, L) 或 (C, L)
- Dropout2d: 输入形状 (N, C, H, W) 或 (N, C, L)
- Dropout3d: 输入形状 (N, C, D, H, W) 或 (C, D, H, W)
- AlphaDropout: 任意形状输入
- FeatureAlphaDropout: 输入形状 (N, C, D, H, W) 或 (C, D, H, W)

## 4. 返回值
- 所有 `forward()` 方法返回 Tensor，形状与输入相同
- 训练时：随机屏蔽元素并缩放输出
- 评估时：直接返回输入（恒等函数）

## 5. 文档要点
- p 必须在 0 到 1 之间（包含边界）
- Dropout2d 对 3D 输入执行 1D 通道 dropout（历史原因）
- AlphaDropout 保持零均值和单位标准差
- 训练/评估模式通过 `self.training` 控制
- 所有操作依赖 `torch.nn.functional` 对应函数

## 6. 源码摘要
- 基类 `_DropoutNd` 验证 p 范围并存储参数
- 所有子类 `forward()` 委托给 `F.dropout*` 函数
- 关键分支：`if p < 0 or p > 1:` 参数验证
- 依赖：`torch.nn.functional` 模块
- 副作用：随机性（Bernoulli 分布）、可能的原地修改

## 7. 示例与用法
**Dropout 示例:**
```python
m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)
```

**Dropout1d 示例:**
```python
m = nn.Dropout1d(p=0.2)
input = torch.randn(20, 16, 32)
output = m(input)
```

## 8. 风险与空白
- 模块包含 6 个独立类，需分别测试
- 缺少具体随机种子控制文档
- Dropout2d 对 3D 输入的行为警告（未来可能改变）
- 未明确支持的 dtype 和设备限制
- AlphaDropout 与 SELU 激活的配合关系需验证
- 缺少性能基准和内存使用说明