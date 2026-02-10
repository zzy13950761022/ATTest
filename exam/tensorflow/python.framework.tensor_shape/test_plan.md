# tensorflow.python.framework.tensor_shape 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch 用于全局状态 _TENSORSHAPE_V2_OVERRIDE 和 tf2.enabled()
- 随机性处理：固定随机种子用于可重复的形状生成测试

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04, CASE_05
- DEFERRED_SET: CASE_06, CASE_07, CASE_08, CASE_09, CASE_10
- group 列表与 active_group_order: G1(Dimension类), G2(TensorShape类), G3(辅助函数)
- 断言分级策略：首轮使用 weak 断言（基本属性检查），最终轮启用 strong 断言（高级属性）
- 预算策略：每个用例 size=S，max_lines=60-75，max_params=4-6

## 3. 数据与边界
- 正常数据集：已知维度值（0, 5, 10），未知维度（None），混合形状
- 边界值：零维度形状 []，完全未知形状 None，大维度值（接近 int 上限）
- 极端形状：嵌套形状，混合已知/未知维度，空输入
- 负例与异常场景：
  - 负维度值引发 ValueError
  - 无效类型引发 TypeError
  - 不支持的运算异常
  - 形状不兼容场景

## 4. 覆盖映射
- TC-01: 覆盖 Dimension 构造与基本属性（需求 5.1）
- TC-02: 覆盖 Dimension 算术运算（需求 5.1）
- TC-03: 覆盖 TensorShape 构造与属性（需求 5.2）
- TC-04: 覆盖形状兼容性检查（需求 5.3）
- TC-05: 覆盖辅助函数正确性（需求 5.5）

尚未覆盖的风险点：
- V1/V2 模式迭代行为差异（需求 5.4）
- 形状合并与连接操作
- 切片和索引操作
- 序列化/反序列化（proto 转换）
- 全局状态管理风险