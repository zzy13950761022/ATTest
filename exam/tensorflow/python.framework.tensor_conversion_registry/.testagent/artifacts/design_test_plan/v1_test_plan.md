# tensorflow.python.framework.tensor_conversion_registry 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用mock隔离全局注册表和缓存，使用fixtures管理测试状态
- 随机性处理：固定随机种子，使用确定性测试数据

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04
- DEFERRED_SET: CASE_05, CASE_06, CASE_07, CASE_08
- group列表：G1（注册功能测试组）、G2（查询功能测试组）
- active_group_order: G1, G2
- 断言分级策略：首轮使用weak断言，最终轮启用strong断言
- 预算策略：size=S/M，max_lines=60-80，max_params=3-5

## 3. 数据与边界
- 正常数据集：自定义类类型、不同优先级值、模拟转换函数
- 边界值：priority=0或负数、空类型元组、None输入
- 极端形状：大量类型注册、高并发访问
- 空输入：查询空类型、注册空函数
- 负例与异常场景：
  - 禁止类型注册（Python数值类型、NumPy类型）
  - 非类型对象作为base_type
  - 无效转换函数签名
  - 转换函数返回NotImplemented
  - as_ref参数处理异常

## 4. 覆盖映射
- TC-01 (CASE_01): 覆盖基本注册与查询功能
- TC-02 (CASE_02): 覆盖禁止类型异常处理
- TC-03 (CASE_03): 覆盖未注册类型查询
- TC-04 (CASE_04): 覆盖优先级排序机制
- TC-05 (CASE_05): 覆盖类型元组注册（deferred）

尚未覆盖的风险点：
- 线程安全性的具体实现验证
- 缓存失效的具体触发条件
- 转换函数异常传播机制
- 大量并发注册的性能影响
- 模块重载对全局状态的影响