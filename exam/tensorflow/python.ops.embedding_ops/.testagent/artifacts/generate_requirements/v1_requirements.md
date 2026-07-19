# tensorflow.python.ops.embedding_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 从嵌入张量中查找和组合嵌入向量
  - 支持密集、稀疏和 RaggedTensor 输入
  - 支持 "mod" 和 "div" 分区策略
  - 支持 "sum"/"mean"/"sqrtn" 组合方式
  - 支持 L2 范数裁剪
- 不在范围内的内容
  - 嵌入张量的训练/优化过程
  - 分布式训练中的梯度同步
  - 自定义嵌入初始化策略

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - params: Tensor/list, 形状 [vocab_size, embedding_dim], 无默认值
  - ids: Tensor (int32/int64), 任意形状, 无默认值
  - max_norm: float/None, 默认 None
  - sp_ids: SparseTensor, 形状任意, 无默认值
  - sp_weights: SparseTensor/None, 默认 None
  - combiner: str, 默认 "mean"
  - default_id: int, 默认 None
  - name: str/None, 默认 None
- 有效取值范围/维度/设备要求
  - ids 值必须在 [0, vocab_size) 范围内
  - 稀疏索引必须按行主序排列
  - 权重形状必须与稀疏索引匹配
  - 支持 CPU/GPU 设备
- 必需与可选组合
  - embedding_lookup_v2: params, ids 必需; max_norm, name 可选
  - embedding_lookup_sparse_v2: params, sp_ids 必需; sp_weights, combiner, max_norm, name 可选
  - safe_embedding_lookup_sparse_v2: 同上，增加 default_id 可选
- 随机性/全局状态要求
  - 无随机性要求
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - embedding_lookup_v2: 形状为 `shape(ids) + shape(params)[1:]` 的密集张量
  - embedding_lookup_sparse_v2: 形状为 `[d0, p1, ..., pm]` 的密集张量
  - safe_embedding_lookup_sparse_v2: 同上，但处理无效 ID
- 容差/误差界（如浮点）
  - float32: 相对误差 1e-5
  - float16/bfloat16: 相对误差 1e-2
  - 组合操作精度: "mean" 和 "sqrtn" 需验证数值稳定性
- 状态变化或副作用检查点
  - 无状态变化
  - 无副作用
  - 设备位置保持不变

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - params 为空列表或空张量
  - ids 包含超出 vocab_size 的值
  - 稀疏索引未按行主序排列
  - 权重与稀疏索引形状不匹配
  - 无效的 combiner 字符串
  - 负的 max_norm 值
- 边界值（空、None、0 长度、极端形状/数值）
  - vocab_size = 0 或 1
  - embedding_dim = 0 或 1
  - ids 为空张量
  - 稀疏张量无值
  - 权重全为 0 或负值
  - 极端大形状 (超过 int32 范围)
  - 极端小/大数值 (inf, nan)

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 运行时环境
  - GPU 设备 (可选)
  - 无网络/文件依赖
- 需要 mock/monkeypatch 的部分
  - `tensorflow.python.ops.array_ops.gather`
  - `tensorflow.python.ops.sparse_ops.sparse_segment_sum`
  - `tensorflow.python.ops.sparse_ops.sparse_segment_mean`
  - `tensorflow.python.ops.sparse_ops.sparse_segment_sqrt_n`
  - `tensorflow.python.framework.ops.colocate_with`
  - `tensorflow.python.ops.math_ops._clip_by_norm`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 密集查找基本功能验证
  2. 稀疏查找三种组合器行为
  3. 分区策略 "mod" vs "div" 差异
  4. L2 范数裁剪边界条件
  5. safe_embedding_lookup_sparse_v2 默认 ID 处理
- 可选路径（中/低优先级合并为一组列表）
  - RaggedTensor 输入支持
  - 多设备 (CPU/GPU) 一致性
  - 不同数值精度 (float16/32/64, bfloat16)
  - 大规模词汇表性能
  - 梯度计算正确性
  - 动态形状支持
  - 嵌套分区策略
- 已知风险/缺失信息（仅列条目，不展开）
  - 类型注解缺失
  - 多实体函数覆盖不全
  - 设备相关异常处理差异
  - 稀疏张量验证逻辑
  - 极端形状内存溢出风险