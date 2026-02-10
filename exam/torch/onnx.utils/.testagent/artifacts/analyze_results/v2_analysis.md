# 测试结果分析

## 状态与统计
- **状态**: 未完全通过
- **通过测试**: 5
- **失败测试**: 10
- **错误测试**: 0
- **收集错误**: 否

## 待修复 BLOCK 列表 (1个)

### 1. HEADER - mock_model_to_graph函数
- **错误类型**: ValueError
- **修复动作**: rewrite_block
- **原因**: mock_model_to_graph函数返回值数量不正确，应返回3个值但只返回了2个。所有失败的测试都源于此问题。

## 延迟处理
- 9个测试因错误类型重复被标记为deferred
- 修复HEADER中的mock函数后，这些测试应能自动通过

## 停止建议
- **stop_recommended**: false
- 问题明确且可修复，不需要停止测试流程