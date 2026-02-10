## 测试结果分析

### 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 6
- **错误**: 0
- **覆盖率**: 49%

### 待修复 BLOCK 列表（3个）

1. **BLOCK_ID**: CASE_01
   - **Action**: rewrite_block
   - **Error Type**: KeyError
   - **问题**: mock调用参数访问方式错误，应检查位置参数而非关键字参数

2. **BLOCK_ID**: CASE_02
   - **Action**: rewrite_block
   - **Error Type**: KeyError
   - **问题**: mock调用参数访问方式错误，应检查位置参数而非关键字参数

3. **BLOCK_ID**: CASE_03
   - **Action**: rewrite_block
   - **Error Type**: KeyError
   - **问题**: mock调用参数访问方式错误，应检查位置参数而非关键字参数

### 停止建议
- **stop_recommended**: false
- **原因**: 所有失败都是相同类型的KeyError，但需要修复mock调用参数的访问方式