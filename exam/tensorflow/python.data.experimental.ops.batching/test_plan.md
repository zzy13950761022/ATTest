# tensorflow.python.data.experimental.ops.batching 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用TensorFlow测试工具，无外部mock需求
- 随机性处理：固定随机种子确保可重复性
- 设备策略：CPU-only测试，避免GPU依赖

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04
- **DEFERRED_SET**: CASE_05, CASE_06
- **group列表**: G1 (dense_to_ragged_batch), G2 (dense_to_sparse_batch)
- **active_group_order**: G1, G2
- **断言分级策略**: 首轮使用weak断言（类型、形状、基本属性），后续启用strong断言（精确值、一致性）
- **预算策略**: 每个用例max_lines≤80，max_params≤6，size=S

## 3. 数据与边界
- **正常数据集**: 不同形状的密集张量，包含标量、向量、矩阵
- **随机生成策略**: 固定种子生成可重复的随机形状和值
- **边界值**: batch_size=1, 空数据集, 极端形状张量
- **极端形状**: 超大维度张量, 零长度维度
- **空输入**: 空数据集处理验证
- **负例场景**: 无效batch_size, 不兼容row_shape, 错误dtype

## 4. 覆盖映射
### TC-01 (CASE_01): dense_to_ragged_batch基本功能
- 需求: 密集张量到RaggedTensor转换
- 约束: batch_size>0, 支持不同形状输入
- 断言: 输出类型, 批大小, 元素形状, row_splits_dtype

### TC-02 (CASE_02): drop_remainder参数行为
- 需求: 验证drop_remainder=True时的批次处理
- 约束: 正确处理不足批大小的剩余元素
- 断言: 批次数量, 剩余处理, 输出形状

### TC-03 (CASE_03): dense_to_sparse_batch基本功能
- 需求: 密集张量到SparseTensor转换
- 约束: 输入元素rank与row_shape匹配
- 断言: 输出类型, 稀疏格式, row_shape合规性

### TC-04 (CASE_04): row_shape约束验证
- 需求: 验证维度大小不超过row_shape
- 约束: 每个维度大小≤row_shape对应维度
- 断言: 形状兼容性, 无错误, 输出结构

### 尚未覆盖的风险点
- 已弃用函数兼容性
- 混合数据类型批处理
- 嵌套结构批处理
- 不同设备行为一致性
- 性能基准测试