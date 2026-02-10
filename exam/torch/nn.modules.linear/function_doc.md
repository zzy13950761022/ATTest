# torch.nn.modules.linear - 函数说明

## 1. 基本信息
- **FQN**: torch.nn.modules.linear
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/nn/modules/linear.py`
- **签名**: 模块（包含多个类）
- **对象类型**: Python 模块

## 2. 功能概述
该模块提供 PyTorch 中的线性变换层。包含 Identity、Linear、Bilinear、LazyLinear 等类。
核心类 Linear 实现 y = xA^T + b 的线性变换。

## 3. 参数说明
**Linear 类构造函数**：
- in_features (int): 输入特征维度，必须为正整数
- out_features (int): 输出特征维度，必须为正整数  
- bias (bool/True): 是否包含偏置项
- device (None): 张量设备
- dtype (None): 张量数据类型

**forward 方法**：
- input (Tensor): 输入张量，形状 (*, in_features)

## 4. 返回值
- Linear.forward(): 返回 Tensor，形状 (*, out_features)
- 所有类继承自 Module，支持 PyTorch 训练流程

## 5. 文档要点
- 输入形状: (*, H_in)，* 表示任意维度数
- 输出形状: (*, H_out)，除最后一维外与输入相同
- 权重形状: (out_features, in_features)
- 偏置形状: (out_features)（当 bias=True）
- 支持 TensorFloat32，特定 ROCm 设备上 float16 使用不同精度

## 6. 源码摘要
- Linear.forward() 调用 F.linear(input, weight, bias)
- reset_parameters() 使用 kaiming_uniform 初始化权重
- 偏置初始化使用均匀分布 U(-bound, bound)，bound = 1/√fan_in
- LazyLinear 延迟初始化，首次 forward 时推断 in_features
- Bilinear 处理两个输入：y = x₁ᵀAx₂ + b

## 7. 示例与用法
```python
>>> m = nn.Linear(20, 30)
>>> input = torch.randn(128, 20)
>>> output = m(input)  # 形状: (128, 30)
```

## 8. 风险与空白
- 模块包含多个类（Linear、Bilinear、Identity、LazyLinear），需分别测试
- 未明确指定输入张量的 dtype 和设备兼容性约束
- 边界情况：in_features=0 或 out_features=0 的行为未说明
- 延迟初始化（LazyLinear）的异常处理未详细描述
- 量化相关类 NonDynamicallyQuantizableLinear 的用途特殊