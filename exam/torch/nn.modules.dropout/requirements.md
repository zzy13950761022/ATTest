# torch.nn.modules.dropout 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证 Dropout 系列层在训练/评估模式下的正确行为，包括随机屏蔽、输出缩放、形状保持
- 不在范围内的内容：Dropout 的理论推导、与其他正则化方法对比、性能基准测试

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - p (float, 默认 0.5)：dropout 概率，范围 [0, 1]
  - inplace (bool, 默认 False)：是否原地操作
- 有效取值范围/维度/设备要求：
  - p ∈ [0, 1]（包含边界）
  - Dropout1d：输入形状 (N, C, L) 或 (C, L)
  - Dropout2d：输入形状 (N, C, H, W) 或 (N, C, L)
  - Dropout3d：输入形状 (N, C, D, H, W) 或 (C, D, H, W)
  - AlphaDropout/FeatureAlphaDropout：特定形状要求
- 必需与可选组合：p 必须提供有效浮点数，inplace 可选
- 随机性/全局状态要求：依赖 torch.random 状态，需控制随机种子

## 3. 输出与判定
- 期望返回结构及关键字段：Tensor，形状与输入完全相同
- 容差/误差界（如浮点）：浮点误差在 1e-6 内，AlphaDropout 需保持零均值和单位标准差
- 状态变化或副作用检查点：
  - 训练模式：输出元素随机置零，非零元素缩放 1/(1-p)
  - 评估模式：恒等映射，输出等于输入
  - inplace=True 时修改输入张量

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - p < 0 或 p > 1 触发 ValueError
  - 不支持的输入形状触发 RuntimeError
  - Dropout2d 对 3D 输入发出警告
- 边界值（空、None、0 长度、极端形状/数值）：
  - p = 0：无 dropout，输出等于输入
  - p = 1：全部置零（训练模式）
  - 空张量、零维度输入
  - 极端形状（超大维度、单元素张量）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：torch 库，CUDA 设备（可选）
- 需要 mock/monkeypatch 的部分：
  - torch.nn.functional.dropout* 函数
  - torch.random 状态
  - self.training 属性切换

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 训练/评估模式切换验证
  2. p 参数边界值测试（0, 0.5, 1）
  3. 各子类形状约束验证
  4. inplace 操作副作用检查
  5. AlphaDropout 统计特性验证
- 可选路径（中/低优先级合并为一组列表）：
  - 不同 dtype 支持（float32, float64）
  - 不同设备支持（CPU, CUDA）
  - 批量大小变化影响
  - 随机种子控制一致性
  - 内存使用和性能检查
- 已知风险/缺失信息（仅列条目，不展开）：
  - Dropout2d 对 3D 输入的特殊行为
  - 随机性测试的统计可靠性
  - 极端形状的性能退化
  - AlphaDropout 与 SELU 激活兼容性
  - 缺少官方性能基准