# tensorflow.python.ops.stateless_random_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 测试重点：确定性验证、参数约束、分布特性

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03
- DEFERRED_SET: CASE_04, CASE_05
- 测试文件路径：tests/test_tensorflow_python_ops_stateless_random_ops.py
- 断言分级策略：首轮使用 weak 断言，最终轮启用 strong 断言
- 预算策略：size=S, max_lines=80, max_params=6

## 3. 数据与边界
- 正常数据集：均匀分布浮点数、整数，正态分布
- 随机生成策略：固定种子确保可重复性
- 边界值：空形状、零维度、极端数值范围
- 极端形状：多维张量、大尺寸张量
- 空输入：shape 为空列表
- 负例场景：种子形状错误、参数不一致、类型不匹配

## 4. 覆盖映射
- TC-01: 浮点数均匀分布核心路径验证
- TC-02: 整数类型均匀分布核心路径验证  
- TC-03: 正态分布核心路径验证（需要 mock）
- TC-04: 种子形状 [2] 强制要求验证
- TC-05: 整数类型 minval/maxval 一致性约束验证

## 5. 尚未覆盖的风险点
- 算法参数 "auto_select" 的设备差异性
- 无符号整数类型与 minval/maxval 互斥性
- XLA 编译环境下的约束验证
- 整数类型的偏差问题（非 2 的幂范围）
- 跨设备一致性测试（CPU/GPU/TPU）