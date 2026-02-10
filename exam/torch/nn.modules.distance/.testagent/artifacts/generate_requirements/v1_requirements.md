# torch.nn.modules.distance 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试PairwiseDistance和CosineSimilarity两个距离计算类的正确性、边界条件和异常处理
- 不在范围内的内容：底层functional函数实现、梯度计算、训练过程、其他距离度量方法

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - PairwiseDistance: p(float/2.0), eps(float/1e-6), keepdim(bool/False)
  - CosineSimilarity: dim(int/1), eps(float/1e-8)
- 有效取值范围/维度/设备要求：
  - p可为负值，需验证负范数行为
  - eps必须为正小值
  - dim必须在输入张量维度范围内
  - 输入张量形状必须匹配dim维度大小
  - 支持CPU/GPU设备
- 必需与可选组合：
  - 两个类都需两个输入张量x1, x2
  - 所有参数都有默认值
- 随机性/全局状态要求：无随机性，无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 两个类都返回Tensor
  - PairwiseDistance输出形状：(N)或()，keepdim=True时为(N,1)或(1)
  - CosineSimilarity输出形状：(*1, *2)，去除dim维度
- 容差/误差界（如浮点）：
  - 浮点计算误差在1e-6范围内
  - eps参数影响数值稳定性
- 状态变化或副作用检查点：
  - 无状态变化
  - 无副作用

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 输入张量类型非torch.Tensor
  - 输入形状不匹配（dim维度大小不同）
  - dim超出张量维度范围
  - eps非正数
  - 输入包含inf/nan值
- 边界值（空、None、0长度、极端形状/数值）：
  - 空张量输入
  - 零向量（全零输入）
  - 极端p值（0, 1, inf, -inf）
  - 极小eps值（接近机器精度）
  - 极大张量形状（内存边界）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - torch库依赖
  - 可选CUDA设备
  - 无网络/文件依赖
- 需要mock/monkeypatch的部分：
  - 底层F.pairwise_distance和F.cosine_similarity函数
  - 设备检测逻辑

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. PairwiseDistance默认参数下的欧氏距离计算
  2. CosineSimilarity默认参数下的余弦相似度计算
  3. 输入形状匹配验证和广播机制
  4. 参数边界值（p=1,2,inf,负值）
  5. 异常输入（类型错误、形状不匹配）处理
- 可选路径（中/低优先级合并为一组列表）：
  - keepdim参数效果验证
  - 不同dtype（float32, float64）支持
  - 设备间一致性（CPU vs GPU）
  - 大规模张量性能测试
  - 梯度计算正确性（如需）
  - 多维度输入处理
  - 广播行为的边界条件
- 已知风险/缺失信息（仅列条目，不展开）：
  - p为负值时的具体行为未明确
  - 极端值（inf/nan）处理策略
  - 广播行为的详细边界条件
  - 输入dtype限制的完整说明
  - 内存使用和性能基准