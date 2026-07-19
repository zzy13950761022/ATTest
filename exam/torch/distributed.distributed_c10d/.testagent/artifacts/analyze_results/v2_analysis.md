## 测试结果分析

### 状态与统计
- **状态**: 失败
- **通过**: 4
- **失败**: 0
- **错误**: 2
- **收集错误**: 否

### 待修复 BLOCK 列表 (2个)

1. **BLOCK_ID**: CASE_03
   - **Action**: rewrite_block
   - **Error Type**: AttributeError
   - **原因**: mock_process_group fixture中backend属性不存在

2. **BLOCK_ID**: CASE_04
   - **Action**: rewrite_block
   - **Error Type**: AttributeError
   - **原因**: mock_process_group fixture中backend属性不存在

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 无