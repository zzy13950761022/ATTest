# tensorflow.python.ops.sort_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 测试范围：sort()和argsort()函数的核心功能
- 设备支持：CPU优先，GPU可选

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03
- DEFERRED_SET: CASE_04, CASE_05
- 测试文件路径：tests/test_tensorflow_python_ops_sort_ops.py
- 断言分级策略：首轮使用weak断言，最终启用strong断言
- 预算策略：S/M size，max_lines 70-90，max_params 5-6
- 迭代策略：首轮3个核心用例，后续修复失败用例，最终启用强断言

## 3. 数据与边界
- 正常数据集：随机正态分布、均匀分布、整数范围
- 边界值：空张量、单元素、极端形状(高维)
- 特殊数值：inf, -inf, NaN（待明确行为）
- 整数边界：int8/int16/int32/int64溢出边界
- 轴边界：-1, 0, 中间轴，超出范围
- 方向：ASCENDING, DESCENDING

## 4. 覆盖映射
| TC_ID | 功能覆盖 | 需求覆盖 | 约束验证 |
|-------|----------|----------|----------|
| TC-01 | 1D浮点排序 | 基本排序功能 | dtype, shape保持 |
| TC-02 | 多维轴排序 | 轴参数处理 | 任意轴排序 |
| TC-03 | 整数降序排序 | 整数类型支持 | 降序方向 |
| TC-04 | argsort索引 | 索引正确性 | int32返回类型 |
| TC-05 | 边界异常 | 错误处理 | 轴边界验证 |

## 5. 尚未覆盖的风险点
- NaN排序行为未明确
- stable参数未实现但保留
- 大整数溢出处理细节
- 非最优化轴排序性能
- GPU设备特定行为
- 混合设备张量排序

## 6. Mock目标
- CASE_04需要mock：tensorflow.python.ops.nn_ops.top_k
- CASE_04需要mock：tensorflow.python.ops.array_ops.transpose  
- CASE_04需要mock：tensorflow.python.ops.math_ops.cast
- 其他用例无需mock，直接测试

## 7. 验证Oracle
- numpy.sort / numpy.argsort 作为参考实现
- 手动计算验证边界情况
- tf.gather验证argsort重建