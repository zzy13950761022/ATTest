# tensorflow.python.ops.ragged.ragged_factory_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03
- DEFERRED_SET: CASE_04, CASE_05
- 测试文件路径：tests/test_tensorflow_python_ops_ragged_ragged_factory_ops.py
- 断言分级策略：首轮使用weak断言，最终轮启用strong断言
- 预算策略：每个CASE size=S，max_lines=80，max_params=6

## 3. 数据与边界
- 正常数据集：简单嵌套列表、浮点数列表、多层嵌套结构
- 随机生成策略：固定种子生成嵌套列表
- 边界值：空列表、单元素列表、不一致嵌套深度
- 极端形状：极大嵌套深度、长列表
- 空输入：[]、[[]]、[[[]]]
- 负例与异常场景：
  - 不一致嵌套深度
  - 无效ragged_rank
  - 类型不兼容
  - 无效row_splits_dtype

## 4. 覆盖映射
| TC_ID | 功能覆盖 | 需求/约束 |
|-------|----------|-----------|
| TC-01 | constant基本功能 | 必测路径1 |
| TC-02 | constant_value基本功能 | 必测路径2 |
| TC-03 | dtype自动推断 | 必测路径3 |
| TC-04 | ragged_rank验证 | 必测路径4 |
| TC-05 | 错误处理 | 必测路径5 |

尚未覆盖的风险点：
- placeholder功能（TensorFlow 1.x特定）
- inner_shape参数组合
- 复杂混合类型输入
- 递归深度限制
- 大尺寸输入性能