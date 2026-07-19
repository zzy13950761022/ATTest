# tensorflow.python.compiler.xla.jit 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock eager execution 检测逻辑，monkeypatch 全局状态
- 随机性处理：不适用（无随机参数）
- 测试模式：必须在 graph execution 模式下运行

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04
- DEFERRED_SET: CASE_05, CASE_06, CASE_07
- group 列表：G1（核心功能），G2（参数功能）
- active_group_order: G1 → G2
- 断言分级策略：首轮使用 weak 断言，最终轮启用 strong 断言
- 预算策略：size=S/M，max_lines=50-80，max_params=3-6

## 3. 数据与边界
- 正常数据：bool 参数 True/False，callable 参数 lambda 函数
- 边界值：compile_ops=None，separate_compiled_gradients=None
- 极端场景：嵌套作用域深度测试
- 负例场景：eager execution 模式异常
- 异常输入：非 bool/非 callable 参数类型错误

## 4. 覆盖映射
| TC_ID | 需求覆盖 | 约束覆盖 | 风险点 |
|-------|----------|----------|--------|
| TC-01 | 基本上下文管理器功能 | graph execution 模式 | 状态恢复验证 |
| TC-02 | eager execution 异常 | 错误处理 | mock 准确性 |
| TC-03 | compile_ops bool 参数 | 参数功能 | 编译行为验证 |
| TC-04 | compile_ops callable 参数 | 条件编译 | callable 调用验证 |

## 5. 尚未覆盖的风险点
- 嵌套作用域组合行为
- 作用域外操作聚类编译
- 多线程环境作用域管理
- 与 tf.function 装饰器组合使用
- 具体编译性能影响（尽力而为特性）