# tensorflow.python.debug.lib.check_numerics_callback 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock op_callbacks.add_op_callback、线程局部状态、日志系统
- 随机性处理：固定随机种子，控制张量生成

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_06, CASE_09 (5个核心用例)
- DEFERRED_SET: CASE_04, CASE_05, CASE_07, CASE_08, CASE_10, CASE_11 (6个延期用例)
- group 列表与 active_group_order: G1, G2, G3
- 断言分级策略：首轮使用weak断言，最终轮启用strong断言
- 预算策略：size=S(60-90行), max_params≤4, 优先非参数化

## 3. 数据与边界
- 正常数据集：浮点张量(32/64位)，正常数值范围
- 边界值：NaN, Infinity, -Infinity浮点值
- 极端形状：小/大参数值(1-100)，零值边界
- 空输入：不适用（无张量输入参数）
- 负例：非浮点数据类型(int32, bool)，无效参数(零/负值)
- 异常场景：TPU环境要求，线程并发，重复调用

## 4. 覆盖映射
| TC-ID | 需求覆盖 | 约束覆盖 |
|-------|----------|----------|
| TC-01 | 基本启用功能 | 参数默认值，副作用检查 |
| TC-02 | NaN检测，异常抛出 | 浮点数据类型，InvalidArgumentError |
| TC-03 | 幂等性 | 线程局部，多次调用 |
| TC-04 | 参数边界 | stack_height_limit, path_length_limit |
| TC-05 | Infinity检测 | 正负无穷值处理 |
| TC-06 | 禁用功能 | 状态清理 |
| TC-07 | 启用禁用循环 | 状态管理 |
| TC-08 | 线程局部 | 线程隔离 |
| TC-09 | 回调类 | CheckNumericsCallback实例 |
| TC-10 | 非浮点忽略 | dtype.is_floating检查 |
| TC-11 | 忽略列表 | IGNORE_OP_OUTPUTS覆盖 |

## 5. 尚未覆盖的风险点
- TPU环境特殊要求（需硬件）
- SAFE_OPS列表完整性验证
- 性能开销量化测试
- 并发环境线程安全性深度测试
- 内存泄漏长期运行验证