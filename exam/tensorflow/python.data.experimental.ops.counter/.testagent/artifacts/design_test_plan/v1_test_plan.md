# tensorflow.python.data.experimental.ops.counter 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：每个测试用例独立创建数据集实例，避免状态污染
- 随机性处理：确定性计数，无需随机种子控制

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04
- DEFERRED_SET: CASE_05
- group 列表与 active_group_order: G1(核心计数功能), G2(数据类型与边界)
- 断言分级策略：首轮使用weak断言（dataset_type, element_shape, dtype_match, sequence_start等）
- 预算策略：size=S, max_lines=60-70, max_params=3

## 3. 数据与边界
- 正常数据集：整数序列（默认参数、指定参数、负步长）
- 边界值：start=0/极大/极小值，step=0/负值/浮点值
- 数据类型边界：int32, int64, float32, float64
- 负例与异常场景：无效dtype、非数值参数、不支持类型

## 4. 覆盖映射
- TC-01: 默认参数创建int64计数数据集（需求1）
- TC-02: 指定start和step参数验证序列生成（需求2）
- TC-03: 不同dtype参数验证（需求3）
- TC-04: 负step递减计数验证（需求4）
- TC-05: 浮点start/step参数验证（需求5）

## 5. 尚未覆盖的风险点
- 极端数值边界（极大/极小start/step）
- step=0的特殊情况
- 与take()操作组合使用验证
- 多次调用创建独立数据集验证
- 数据类型转换验证