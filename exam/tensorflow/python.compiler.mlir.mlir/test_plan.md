# tensorflow.python.compiler.mlir.mlir 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures 隔离底层 pywrap_mlir 调用
- 随机性处理：固定随机种子控制 TensorFlow 图构造

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04
- DEFERRED_SET: CASE_05, CASE_06
- group 列表与 active_group_order: G1 (convert_graph_def), G2 (convert_function)
- 断言分级策略：首轮使用 weak 断言（返回字符串、包含模块、无异常）
- 预算策略：size=S, max_lines=70-75, max_params=3-4

## 3. 数据与边界
- 正常数据集：简单计算图、加法函数、自定义 pipeline
- 边界值：无效 GraphDef、无效 ConcreteFunction、空字符串 pipeline
- 负例与异常场景：无效输入触发 InvalidArgumentError

## 4. 覆盖映射
- TC-01: 覆盖 convert_graph_def 基本功能需求
- TC-02: 覆盖 pass_pipeline 和 show_debug_info 参数验证
- TC-03: 覆盖 convert_function 基本功能需求  
- TC-04: 覆盖函数转换的参数验证
- TC-05: 覆盖异常处理场景
- TC-06: 覆盖函数异常处理场景

## 5. 尚未覆盖的风险点
- 底层 pywrap_mlir 实现细节未知
- pass_pipeline 格式规范缺失
- 输出格式详细规范不明确
- 复杂图结构转换验证不足