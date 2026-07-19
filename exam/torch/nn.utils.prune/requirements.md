# torch.nn.utils.prune 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证 PyTorch 剪枝模块的正确性，包括非结构化/结构化/全局剪枝方法，确保掩码计算、参数修改、剪枝移除等功能符合预期
- 不在范围内的内容：自定义剪枝算法实现、剪枝后模型性能评估、硬件加速优化

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - amount: int/float，剪枝量，int 表示绝对数量，float 表示比例
  - dim: int，结构化剪枝的通道维度，默认 -1
  - importance_scores: Tensor，自定义重要性分数，可选
  - pruning_method: BasePruningMethod 子类，剪枝方法类
  - parameters_to_prune: tuple/list，全局剪枝的参数列表

- 有效取值范围/维度/设备要求：
  - amount: float ∈ [0.0, 1.0]，int ≥ 0 且 ≤ 参数总数
  - 结构化剪枝要求张量至少 2 维
  - 支持 CPU/CUDA 设备
  - 全局剪枝仅支持 unstructured 类型

- 必需与可选组合：
  - 结构化剪枝必须提供 dim 参数
  - importance_scores 可选，形状需匹配参数
  - 全局剪枝必须提供 parameters_to_prune 列表

- 随机性/全局状态要求：
  - RandomUnstructured/RandomStructured 包含随机性
  - 需要设置随机种子保证可重复性
  - 剪枝状态存储在模块缓冲区中

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 工具函数返回修改后的 nn.Module
  - 剪枝类实例包含 apply() 方法
  - 模块添加 _forward_pre_hooks 和缓冲区

- 容差/误差界（如浮点）：
  - 浮点剪枝比例容差：±1e-6
  - 掩码计算正确率：100%
  - 参数修改一致性检查

- 状态变化或副作用检查点：
  - 参数张量被掩码修改
  - 模块缓冲区存储掩码和原始参数
  - 前向钩子注册正确
  - remove() 后缓冲区清理但参数保持剪枝状态

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - amount 超出范围：ValueError
  - 非数值 amount 类型：TypeError
  - 结构化剪枝应用于 1D 张量：ValueError
  - 无效参数名：AttributeError
  - importance_scores 形状不匹配：RuntimeError

- 边界值（空、None、0 长度、极端形状/数值）：
  - amount=0：无剪枝但状态正确
  - amount=1.0 或全部参数：完全剪枝
  - 空参数列表：ValueError
  - None 输入：TypeError
  - 极端大形状张量：内存检查

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - PyTorch 库依赖
  - CUDA 设备（可选）
  - 无网络/文件依赖

- 需要 mock/monkeypatch 的部分：
  - `torch.rand` 用于随机剪枝测试
  - `torch.nn.Module._forward_pre_hooks` 注册验证
  - `torch.Tensor.__getitem__` 用于掩码应用检查
  - `torch.norm` 用于 LnStructured 剪枝

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基本非结构化剪枝：amount 为 float 和 int 的正确掩码计算
  2. 结构化剪枝：dim 参数正确选择通道，L1/L2 范数计算
  3. 全局剪枝：多参数统一剪枝，重要性分数跨参数比较
  4. 剪枝移除：remove() 清理缓冲区但保留剪枝效果
  5. 边界条件：amount=0、amount=全部参数、无效输入异常

- 可选路径（中/低优先级合并为一组列表）：
  - 迭代剪枝组合（PruningContainer）
  - 自定义 importance_scores 优先级
  - 不同设备（CPU/CUDA）一致性
  - 大模型内存使用监控
  - 梯度计算正确性（剪枝参数梯度为0）
  - 序列化/反序列化后剪枝状态保持

- 已知风险/缺失信息（仅列条目，不展开）：
  - 全局剪枝仅支持 unstructured 类型限制
  - 缺少完整类型注解
  - 结构化剪枝 dim 默认值 -1 的语义
  - 迭代剪枝组合的复杂行为
  - 剪枝后模型训练稳定性