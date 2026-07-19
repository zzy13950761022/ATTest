# tensorflow.python.ops.candidate_sampling_ops 测试需求

## 1. 目标与范围
- **主要功能与期望行为**：验证候选采样操作模块的6个主要函数（uniform_candidate_sampler、log_uniform_candidate_sampler、learned_unigram_candidate_sampler、fixed_unigram_candidate_sampler、all_candidate_sampler、compute_accidental_hits）能正确从类别空间采样负样本，返回符合概率分布的三元组结果，支持不同采样策略和唯一性约束。
- **不在范围内的内容**：底层C++实现细节、TensorFlow框架核心随机数生成器、GPU/TPU设备特定优化、分布式训练场景、动态图与静态图模式差异。

## 2. 输入与约束
- **参数列表**：
  - `true_classes`: int64类型，形状`[batch_size, num_true]`
  - `num_true`: int类型，每个训练样本的目标类别数
  - `num_sampled`: int类型，采样类别数
  - `unique`: bool类型，是否采样唯一类别
  - `range_max`: int类型，可能类别总数
  - `seed`: int类型，随机种子
  - `name`: string类型，操作名称
- **有效取值范围/维度/设备要求**：
  - `true_classes`值域：`[0, range_max-1]`
  - `num_sampled` ≥ 1
  - `range_max` ≥ 1
  - 当`unique=True`时，`num_sampled` ≤ `range_max`
  - 支持CPU设备，无需特殊硬件
- **必需与可选组合**：
  - `true_classes`、`num_true`、`num_sampled`、`range_max`为必需参数
  - `unique`、`seed`、`name`为可选参数
- **随机性/全局状态要求**：
  - 依赖随机数生成器，需通过`seed`参数控制
  - 无全局状态依赖

## 3. 输出与判定
- **期望返回结构及关键字段**：
  - `sampled_candidates`: int64张量，形状`[num_sampled]`，采样类别ID
  - `true_expected_count`: float张量，形状同`true_classes`，真实类别期望计数
  - `sampled_expected_count`: float张量，形状同`sampled_candidates`，采样类别期望计数
- **容差/误差界**：
  - 浮点误差：相对误差≤1e-6，绝对误差≤1e-8
  - 概率分布验证：采样频率与理论概率误差≤0.01（大样本）
- **状态变化或副作用检查点**：
  - 无文件I/O
  - 无网络访问
  - 无全局变量修改

## 4. 错误与异常场景
- **非法输入/维度/类型触发的异常或警告**：
  - `true_classes`非int64类型
  - `num_sampled` > `range_max`且`unique=True`
  - `range_max` ≤ 0
  - `num_sampled` ≤ 0
  - `true_classes`值超出`[0, range_max-1]`范围
- **边界值**：
  - `range_max` = 1
  - `num_sampled` = 1
  - `batch_size` = 1
  - `num_true` = 1
  - `true_classes`为空张量
  - `seed` = 0（特殊种子值）

## 5. 依赖与环境
- **外部资源/设备/网络/文件依赖**：
  - TensorFlow运行时环境
  - 无网络/文件系统依赖
- **需要mock/monkeypatch的部分**：
  - `tensorflow.python.ops.gen_candidate_sampling_ops.uniform_candidate_sampler`
  - `tensorflow.python.ops.gen_candidate_sampling_ops.log_uniform_candidate_sampler`
  - `tensorflow.python.ops.gen_candidate_sampling_ops.learned_unigram_candidate_sampler`
  - `tensorflow.python.ops.gen_candidate_sampling_ops.fixed_unigram_candidate_sampler`
  - `tensorflow.python.ops.gen_candidate_sampling_ops.all_candidate_sampler`
  - `tensorflow.python.ops.gen_candidate_sampling_ops.compute_accidental_hits`
  - `tensorflow.python.framework.random_seed.get_seed`
  - `tensorflow.python.ops.random_ops.random_uniform`
  - `tensorflow.python.ops.random_ops.random_shuffle`
  - `tensorflow.python.ops.array_ops.reshape`
  - `tensorflow.python.ops.array_ops.expand_dims`

## 6. 覆盖与优先级
- **必测路径（高优先级）**：
  1. 验证uniform_candidate_sampler均匀分布采样正确性
  2. 测试unique=True时num_sampled≤range_max约束
  3. 验证compute_accidental_hits意外命中计算
  4. 测试不同分布类型（均匀、对数均匀、学习分布）差异
  5. 验证随机种子控制可重现性
- **可选路径（中/低优先级）**：
  - 大规模range_max（>10000）性能测试
  - 批量大小batch_size极端值测试
  - 混合精度类型兼容性
  - 不同TensorFlow版本兼容性
  - 多线程并发安全性
- **已知风险/缺失信息**：
  - 底层C++操作实现细节未知
  - 特定分布参数（如unigram分布）默认值
  - GPU/TPU设备差异
  - 动态图模式行为差异