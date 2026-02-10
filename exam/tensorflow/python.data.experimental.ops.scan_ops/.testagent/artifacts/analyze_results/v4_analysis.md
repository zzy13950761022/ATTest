## 测试结果分析

### 状态与统计
- **状态**: 未完全通过
- **通过**: 1 个测试
- **失败**: 5 个测试
- **错误**: 0 个错误
- **覆盖率**: 35%

### 待修复 BLOCK 列表 (3个)

1. **BLOCK_ID**: CASE_03
   - **Action**: adjust_assertion
   - **Error Type**: AssertionError
   - **原因**: 测试期望弃用警告但未收到，需要调整断言逻辑

2. **BLOCK_ID**: CASE_04 (测试用例1)
   - **Action**: adjust_assertion
   - **Error Type**: AssertionError
   - **原因**: 测试期望弃用警告但未收到，需要调整断言逻辑

3. **BLOCK_ID**: CASE_04 (测试用例2)
   - **Action**: adjust_assertion
   - **Error Type**: AssertionError
   - **原因**: 测试期望弃用警告但未收到，需要调整断言逻辑

### 延迟处理
- 2个测试因错误类型重复被标记为deferred

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 无