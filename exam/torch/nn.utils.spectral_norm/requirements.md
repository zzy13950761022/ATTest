# torch.nn.utils.spectral_norm 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 对模块的指定权重参数应用谱归一化
  - 通过幂迭代法计算权重矩阵的谱范数
  - 添加前向传播钩子实现动态权重重缩放
  - 返回带有谱归一化钩子的原始模块
- 不在范围内的内容
  - 替代函数 `torch.nn.utils.parametrizations.spectral_norm` 的功能
  - 谱归一化的数学原理验证
  - 训练过程中的收敛性测试

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `module`: nn.Module，必需，包含权重参数的模块
  - `name`: str，默认'weight'，权重参数名称
  - `n_power_iterations`: int，默认1，幂迭代次数
  - `eps`: float，默认1e-12，数值稳定性epsilon
  - `dim`: Optional[int]，默认None，对应输出数量的维度
- 有效取值范围/维度/设备要求
  - `module` 必须包含名为 `name` 的参数
  - 权重张量维度 ≥ 2
  - `n_power_iterations` ≥ 0
  - `eps` > 0
  - `dim` 为 None 或有效维度索引
- 必需与可选组合
  - `module` 为必需参数
  - 其他参数均有默认值，可选
- 随机性/全局状态要求
  - 幂迭代算法包含随机初始化
  - 模块状态被修改（添加钩子）

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回原始模块类型实例
  - 模块包含 `weight_u` 和 `weight_v` 缓冲区
  - 模块具有前向传播钩子
- 容差/误差界（如浮点）
  - 谱范数计算误差在 `eps` 范围内
  - 浮点计算使用默认精度容差
- 状态变化或副作用检查点
  - 模块添加了 `_forward_pre_hooks`
  - 权重参数被包装为 `SpectralNorm` 实例
  - 缓冲区 `weight_u` 和 `weight_v` 正确初始化

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 模块不包含指定参数名时抛出 AttributeError
  - 权重张量维度 < 2 时抛出 ValueError
  - `n_power_iterations` 为负数时抛出 ValueError
  - `eps` ≤ 0 时抛出 ValueError
  - `dim` 超出权重张量维度范围时抛出 IndexError
- 边界值（空、None、0 长度、极端形状/数值）
  - `n_power_iterations = 0`（无迭代）
  - `eps` 极小值（如 1e-30）
  - 权重值为零矩阵
  - 权重值为单位矩阵
  - 极端形状（如 1×1 或超大矩阵）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - PyTorch 库依赖
  - CUDA 设备支持（可选）
- 需要 mock/monkeypatch 的部分
  - `torch.nn.utils.spectral_norm.SpectralNorm.apply` 方法
  - 模块的 `_parameters` 字典访问
  - 随机数生成器（用于幂迭代初始化）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 标准线性层谱归一化
  2. ConvTranspose 模块的特殊 dim 处理
  3. 自定义参数名的谱归一化
  4. 多轮幂迭代验证
  5. 异常参数输入的错误处理
- 可选路径（中/低优先级合并为一组列表）
  - 不同设备（CPU/GPU）的兼容性
  - 各种模块类型（Conv1d/2d/3d, Linear, etc.）
  - 不同权重形状的边界情况
  - 与其他 PyTorch 功能的集成
  - 前向传播钩子的正确触发
- 已知风险/缺失信息（仅列条目，不展开）
  - 函数已标记为弃用
  - 未明确支持的模块类型限制
  - `n_power_iterations` 范围约束不明确
  - `eps` 值的有效范围未定义
  - 异常情况的详细说明缺失