## 测试结果分析

### 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 0
- **错误**: 3
- **收集错误**: 否

### 待修复 BLOCK 列表 (1个)
1. **BLOCK_ID**: HEADER
   - **Action**: fix_dependency
   - **Error Type**: AttributeError
   - **原因**: tensorflow.python模块路径不存在，需要修正mock_context fixture中的导入路径

### 延迟处理
- CASE_02: 依赖HEADER块的mock_context fixture，等待HEADER修复
- CASE_05: 依赖HEADER块的mock_context fixture，等待HEADER修复

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 无