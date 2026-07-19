# tensorflow.python.data.experimental.ops.resampling 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock tf.data.Dataset.rejection_resample 方法，monkeypatch 弃用警告检查
- 随机性处理：固定随机种子，控制 RNG 用于确定性测试

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01（基本功能验证）、CASE_02（弃用警告验证）、CASE_03（分布调整验证）
- DEFERRED_SET: CASE_04（可选参数initial_dist）、CASE_05（边界值测试）
- group 列表与 active_group_order: G1（核心功能验证）、G2（分布调整与随机性）
- 断言分级策略：首轮使用 weak 断言（基本功能验证），后续启用 strong 断言（统计验证）
- 预算策略：size=S/M（70-85行），max_params=4-6，优先保证 smoke_set 可运行

## 3. 数据与边界
- 正常数据集：随机生成类别标签，大小 50-1000 个元素
- 边界值：num_classes=1（单类别），dataset_size=10（小数据集）
- 极端形状：均匀分布 vs 偏斜分布，全零初始分布
- 空输入：空数据集（需验证异常处理）
- 负例：class_func 返回越界值，非浮点分布张量
- 异常场景：形状不匹配，非一维分布，无效种子值

## 4. 覆盖映射
- TC-01 → 需求1：基本功能验证，约束：class_func 映射正确
- TC-02 → 需求4：弃用警告触发，约束：正确标记已弃用
- TC-03 → 需求3：分布调整效果，约束：采样比例误差±5%
- TC-04 → 需求5：可选参数处理，约束：initial_dist=None 时实时估计
- TC-05 → 需求2：边界情况处理，约束：极端值正确处理

## 5. 尚未覆盖的风险点
- 大规模数据集内存使用模式
- 多设备（CPU/GPU）兼容性差异
- 与 tf.data 其他转换组合的副作用
- 分布归一化要求的明确性
- 性能影响未量化（采样丢弃比例）