# torch.nn.utils.clip_grad 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - `clip_grad_norm_`: 计算并原地裁剪参数梯度范数，防止梯度爆炸，返回裁剪前总范数
  - `clip_grad_value_`: 原地裁剪梯度值到 [-clip_value, clip_value] 范围
  - `clip_grad_norm`: 已弃用包装函数，调用 `clip_grad_norm_`
- 不在范围内的内容
  - 梯度计算本身（由反向传播负责）
  - 优化器更新逻辑
  - 分布式训练场景

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `parameters`: Tensor 或 Iterable[Tensor]，任意形状，无默认值
  - `max_norm`: float/int，正数，无默认值
  - `norm_type`: float/int，默认 2.0，支持 `inf` 表示无穷范数
  - `error_if_nonfinite`: bool，默认 False
  - `clip_value`: float/int，正数，无默认值
- 有效取值范围/维度/设备要求
  - `max_norm` > 0
  - `clip_value` > 0
  - `norm_type` 支持 1, 2, inf 等常见范数
  - 支持 CPU 和 CUDA 设备
  - 支持不同 dtype (float32, float64)
- 必需与可选组合
  - `parameters` 必需，可为单个张量或列表
  - `max_norm`/`clip_value` 必需
  - `norm_type` 和 `error_if_nonfinite` 可选
- 随机性/全局状态要求
  - 无随机性
  - 原地修改梯度，影响后续优化器更新

## 3. 输出与判定
- 期望返回结构及关键字段
  - `clip_grad_norm_`: 返回 torch.Tensor 标量，裁剪前总范数
  - `clip_grad_value_`: 返回 None
  - 无梯度参数时返回 torch.tensor(0.)
- 容差/误差界（如浮点）
  - 范数计算容差：相对误差 1e-5
  - 裁剪系数计算：max_norm / (total_norm + 1e-6)
  - 裁剪系数限制在 [0, 1] 范围内
- 状态变化或副作用检查点
  - 梯度张量原地修改
  - 裁剪后梯度值在预期范围内
  - 梯度范数不超过 max_norm
  - 梯度值在 [-clip_value, clip_value] 内

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - `parameters` 为空列表或无梯度参数
  - `max_norm` <= 0 或 `clip_value` <= 0
  - `parameters` 类型错误（非张量或可迭代）
  - `norm_type` 不支持的值
  - `error_if_nonfinite=True` 且梯度范数非有限（NaN/inf）
  - `clip_grad_norm` 调用产生弃用警告
- 边界值（空、None、0 长度、极端形状/数值）
  - 空梯度列表
  - 零范数梯度
  - 极大/极小梯度值
  - 极端形状（大张量、高维张量）
  - `norm_type=inf` 边界情况

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - CUDA 设备（可选）
  - 无网络/文件依赖
- 需要 mock/monkeypatch 的部分
  - `torch.norm` 用于范数计算
  - `torch.clamp` 用于限制裁剪系数
  - `torch.stack` 用于堆叠张量
  - `warnings` 模块用于弃用警告

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. `clip_grad_norm_` 基本功能：单张量梯度裁剪
  2. `clip_grad_value_` 基本功能：梯度值裁剪到指定范围
  3. 多参数列表处理：Iterable[Tensor] 输入
  4. 不同范数类型：norm_type=1, 2, inf
  5. 非有限梯度处理：error_if_nonfinite 两种状态
- 可选路径（中/低优先级合并为一组列表）
  - 已弃用函数 `clip_grad_norm` 的警告行为
  - 不同设备（CPU/CUDA）兼容性
  - 不同 dtype（float32/float64）精度
  - 极端形状和大张量性能
  - 零范数和极小梯度边界
  - 无梯度参数的特殊返回
- 已知风险/缺失信息（仅列条目，不展开）
  - `error_if_nonfinite` 默认值未来会变化
  - 分布式训练场景未覆盖
  - 梯度稀疏性处理未明确
  - 内存使用峰值未定义
  - 线程安全性未说明