# torch.jit._trace 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures 用于内部追踪机制
- 随机性处理：固定随机种子控制张量生成

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_05, CASE_08 (4个核心用例)
- DEFERRED_SET: CASE_03, CASE_04, CASE_06, CASE_07, CASE_09, CASE_10, CASE_11, CASE_12, CASE_13
- group 列表与 active_group_order: G1, G2, G3, G4
- 断言分级策略：首轮使用weak断言，最终轮启用strong断言
- 预算策略：S级用例≤80行，M级用例≤90行，参数≤7个

## 3. 数据与边界
- 正常数据集：随机生成张量，固定种子确保可复现
- 边界值：空张量([0,5])、极端形状、不同dtype(float16/32/64)
- 负例与异常场景：非张量输入、动态控制流、不支持的Python特性

## 4. 覆盖映射
| TC ID | 需求覆盖 | 约束覆盖 |
|-------|----------|----------|
| TC-01 | 基本函数追踪 | 张量操作、严格模式 |
| TC-02 | 多输入函数 | 嵌套元组支持 |
| TC-05 | 模块追踪 | nn.Module支持 |
| TC-08 | 验证机制 | check_trace参数 |
| TC-03 | dtype支持 | 浮点精度处理 |
| TC-04 | 设备支持 | CPU/GPU兼容性 |
| TC-09 | 严格模式 | 可变容器处理 |
| TC-11 | 边界形状 | 空张量处理 |
| TC-12 | 异常输入 | 错误处理机制 |
| TC-13 | 控制流限制 | 动态控制流拒绝 |

## 5. 尚未覆盖的风险点
- 内部参数文档缺失（_force_outplace, _module_class）
- 复杂嵌套结构追踪支持不明确
- 训练/评估模式切换行为
- 非确定性操作处理机制