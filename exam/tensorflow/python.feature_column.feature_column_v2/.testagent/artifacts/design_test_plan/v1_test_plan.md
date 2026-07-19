# tensorflow.python.feature_column.feature_column_v2 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04, CASE_05
- DEFERRED_SET: CASE_06, CASE_07, CASE_08
- group 列表与 active_group_order: G1(核心工厂函数), G2(特征转换函数)
- 断言分级策略：首轮使用weak断言，最终轮启用strong断言
- 预算策略：size=S/M, max_lines=60-80, max_params=3-6

## 3. 数据与边界
- 正常数据集：标准特征列参数组合
- 随机生成策略：固定种子生成测试数据
- 边界值：空key、零形状、极大词汇表、极端边界值
- 极端形状：多维张量形状、大维度嵌入
- 空输入：空词汇表、空边界列表
- 负例与异常场景：
  - 无效key（空字符串/None）
  - 非法形状（负值/零）
  - 非严格递增边界
  - 类型不匹配参数
  - default_value和num_oov_buckets冲突

## 4. 覆盖映射
| TC_ID | 对应需求 | 覆盖约束 |
|-------|----------|----------|
| TC-01 | numeric_column基础创建 | 参数验证、对象属性 |
| TC-02 | categorical_column_with_vocabulary_list基础创建 | 词汇表处理、默认值 |
| TC-03 | bucketized_column边界分桶 | 边界逻辑、分桶正确性 |
| TC-04 | embedding_column维度验证 | 维度检查、初始化 |
| TC-05 | numeric_column错误处理 | 异常场景、错误消息 |

## 5. 尚未覆盖的风险点
- 类型注解不完整导致的运行时错误
- 实验性API稳定性问题
- 张量形状兼容性边界模糊
- 默认值处理逻辑复杂场景
- 大词汇表性能问题