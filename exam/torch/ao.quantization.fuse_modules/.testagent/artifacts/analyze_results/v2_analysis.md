## 测试结果分析

### 状态与统计
- **状态**: 未完全通过
- **通过**: 4个测试
- **失败**: 5个测试
- **错误**: 0个
- **覆盖率**: 80%

### 待修复 BLOCK 列表 (3个)

1. **BLOCK_ID**: CASE_07
   - **Action**: adjust_assertion
   - **Error Type**: AssertionError
   - **原因**: 异常消息不包含type关键词，实际为属性访问错误

2. **BLOCK_ID**: CASE_07 (重复测试)
   - **Action**: adjust_assertion  
   - **Error Type**: AssertionError
   - **原因**: 异常消息不包含type关键词，实际为属性访问错误

3. **BLOCK_ID**: CASE_08
   - **Action**: rewrite_block
   - **Error Type**: AssertionError
   - **原因**: 空列表输入导致get_fuser_method断言失败，需要处理边界情况

### 延迟处理
- test_non_module_type_input[invalid_model3]: 错误类型重复，跳过该块
- test_non_module_type_input[invalid_model4]: 错误类型重复，跳过该块

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 无