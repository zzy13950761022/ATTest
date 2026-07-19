# tensorflow.python.framework.importer 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：每个测试用例使用独立的图上下文，mock C API 调用
- 随机性处理：固定随机种子，使用确定性图结构生成
- 图状态管理：每个测试后清理默认图，避免状态污染

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基本导入）、CASE_02（输入重映射）、CASE_03（返回元素）、CASE_06（异常处理）
- **DEFERRED_SET**: CASE_04（组合功能）、CASE_05（producer_op_list）、CASE_07（无效输入映射）、CASE_08（无效返回元素）、CASE_09（边界值）
- **group 列表**: G1（核心导入功能）、G2（边界与异常处理）
- **active_group_order**: G1 → G2
- **断言分级策略**: 首轮使用 weak 断言（基本验证），最终轮启用 strong 断言（详细验证）
- **预算策略**: 
  - S 型用例：≤70 行，≤5 参数
  - M 型用例：≤85 行，≤6 参数  
  - L 型用例：≤100 行，≤7 参数

## 3. 数据与边界
- **正常数据集**: 简单常量图、占位符图、多操作图、复杂嵌套图
- **随机生成策略**: 确定性图结构生成，固定随机种子
- **边界值**: 
  - 空 GraphDef 导入
  - 空 input_map 字典
  - 空 return_elements 列表
  - 空字符串名称前缀
  - 控制输入映射（^前缀）
  - 张量名称格式边界
- **负例与异常场景**:
  - 无效 GraphDef proto 类型
  - input_map 键不存在于 graph_def
  - return_elements 名称不存在
  - 非字符串类型的 return_elements 元素
  - 格式错误的 proto 数据

## 4. 覆盖映射
| TC ID | 需求/约束 | 优先级 | 状态 |
|-------|-----------|--------|------|
| TC-01 | 基本 GraphDef 导入 | High | SMOKE |
| TC-02 | 带 input_map 的导入 | High | SMOKE |
| TC-03 | 带 return_elements 的导入 | High | SMOKE |
| TC-04 | 组合功能测试 | Medium | DEFERRED |
| TC-05 | producer_op_list 参数 | Low | DEFERRED |
| TC-06 | 无效 GraphDef 异常 | High | SMOKE |
| TC-07 | 无效 input_map 键异常 | Medium | DEFERRED |
| TC-08 | 无效 return_elements 名称异常 | Medium | DEFERRED |
| TC-09 | 边界值测试 | Low | DEFERRED |

## 5. 尚未覆盖的风险点
- producer_op_list 参数的具体业务逻辑（文档不明确）
- 大规模图导入的性能影响
- 并发导入场景的线程安全性
- 辅助函数 import_graph_def_for_function 未覆盖
- 控制输入映射（^前缀名称）的详细验证