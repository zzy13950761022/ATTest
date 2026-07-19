# torch.nn.modules.batchnorm - 函数说明

## 1. 基本信息
- **FQN**: torch.nn.modules.batchnorm
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py`
- **签名**: 模块（包含多个类）
- **对象类型**: Python 模块

## 2. 功能概述
该模块提供批量归一化（Batch Normalization）层的实现。包含 BatchNorm1d、BatchNorm2d、BatchNorm3d 及其懒加载版本，以及 SyncBatchNorm。通过归一化输入数据加速深度网络训练，减少内部协变量偏移。

## 3. 参数说明
核心类 `_BatchNorm` 参数：
- num_features (int): 输入特征数/通道数 C
- eps (float=1e-5): 数值稳定性分母项
- momentum (float=0.1): 运行统计量更新动量，可为 None（累积移动平均）
- affine (bool=True): 是否启用可学习仿射参数
- track_running_stats (bool=True): 是否跟踪运行统计量
- device/dtype: 张量设备与数据类型

## 4. 返回值
- 各 BatchNorm 类实例：返回归一化后的张量，形状与输入相同
- 前向传播：返回 Tensor，形状与输入一致

## 5. 文档要点
- 输入形状约束：
  - BatchNorm1d: (N, C) 或 (N, C, L)
  - BatchNorm2d: (N, C, H, W)
  - BatchNorm3d: (N, C, D, H, W)
  - SyncBatchNorm: 至少 2D 输入
- 训练/评估模式行为不同
- 运行统计量在训练时更新，评估时使用
- 标准差使用有偏估计器（torch.var(input, unbiased=False)）

## 6. 源码摘要
- 关键类层次：`_NormBase` → `_BatchNorm` → 具体维度类
- 前向传播调用 `F.batch_norm` 函数
- 训练模式：使用小批量统计量，更新运行统计量
- 评估模式：使用运行统计量（若存在）
- 懒加载类：延迟初始化 num_features
- SyncBatchNorm：分布式同步统计量

## 7. 示例与用法
```python
# BatchNorm1d 示例
m = nn.BatchNorm1d(100)
input = torch.randn(20, 100)
output = m(input)

# BatchNorm2d 示例
m = nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)
```

## 8. 风险与空白
- 目标为模块而非单一函数，包含 7 个主要类
- 需要测试多个维度变体（1D/2D/3D）
- 需覆盖训练/评估模式切换
- 需测试 affine=False 和 track_running_stats=False 场景
- SyncBatchNorm 需要分布式环境测试
- 懒加载类的延迟初始化行为需验证
- 输入形状验证逻辑需边界测试
- 动量参数为 None 时的累积平均行为
- 数值稳定性（eps）的边界情况