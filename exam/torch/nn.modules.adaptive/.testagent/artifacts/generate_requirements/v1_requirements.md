# torch.nn.modules.adaptive 测试需求

## 1. 目标与范围
- 验证 AdaptiveLogSoftmaxWithLoss 类的高效 softmax 近似算法
- 测试标签分区到多个聚类的正确性和性能优化
- 确保处理大规模输出空间时计算效率
- 不在范围内：其他自适应 softmax 实现、自定义聚类算法

## 2. 输入与约束
- **in_features** (int): 正整数，输入特征维度
- **n_classes** (int): 正整数，类别总数，≥2
- **cutoffs** (Sequence[int]): 唯一正整数的递增序列，每个值在 1 到 n_classes-1 之间
- **div_value** (float, 默认 4.0): 正浮点数，计算聚类大小的指数值
- **head_bias** (bool, 默认 False): 布尔值，控制头部偏置项
- 输入形状：(N, in_features) 或 (in_features)，支持批处理
- 目标形状：(N) 或 ()，每个值满足 0 ≤ target[i] ≤ n_classes-1
- 标签必须按频率排序：索引 0 为最频繁标签

## 3. 输出与判定
- 返回 `_ASMoutput` 命名元组，包含：
  - **output**: 大小为 N 的张量，每个示例的目标对数概率
  - **loss**: 标量，负对数似然损失
- 浮点容差：相对误差 ≤ 1e-5，绝对误差 ≤ 1e-8
- 状态变化：无全局状态修改，仅计算输出
- 副作用：无文件/网络操作，仅内存计算

## 4. 错误与异常场景
- cutoffs 非递增序列 → ValueError
- cutoffs 包含重复值 → ValueError
- cutoffs 值超出 [1, n_classes-1] 范围 → ValueError
- n_classes < 2 → ValueError
- in_features ≤ 0 → ValueError
- 输入/目标形状不匹配 → RuntimeError
- 目标值超出 [0, n_classes-1] 范围 → IndexError
- 边界值：空 cutoffs 列表、极端形状 (0, in_features)、极大 n_classes
- 数值边界：div_value ≤ 0、极大/极小浮点数输入

## 5. 依赖与环境
- 外部依赖：torch.nn.Linear、ModuleList、Sequential
- 设备要求：CPU 和 CUDA 兼容性测试
- 数据类型：float32、float64 支持
- 需要 mock：无外部 API 调用
- 需要 monkeypatch：无动态导入或环境变量

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. 基本功能：标准 cutoffs 配置的前向传播
  2. 参数验证：cutoffs 递增性、唯一性、范围检查
  3. 形状兼容：批处理与非批处理输入
  4. 辅助方法：log_prob 和 predict 的正确性
  5. 设备迁移：CPU ↔ GPU 数据一致性

- 可选路径（中/低优先级）：
  - 不同 div_value 值的影响测试
  - 极端 cutoffs 配置（如单元素列表）
  - 大规模 n_classes 性能测试
  - 混合精度训练兼容性
  - 梯度计算正确性验证
  - 内存使用效率测试

- 已知风险/缺失信息：
  - 文档缺少具体数值示例代码
  - 标签频率排序的实际验证方法
  - 聚类分配算法的内部实现细节
  - 性能基准测试数据缺失
  - 多 GPU 分布式训练支持情况