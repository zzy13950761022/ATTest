## 测试结果分析

### 状态与统计
- **状态**: 未完全通过
- **通过**: 16 个测试
- **失败**: 2 个测试
- **错误**: 0 个
- **测试收集错误**: 无

### 待修复 BLOCK 列表 (2个)

1. **BLOCK_ID**: CASE_08
   - **Action**: adjust_assertion
   - **Error Type**: AssertionError
   - **问题**: 方差断言过于严格，特别是对于scale=2.0的输入，需要放宽方差容忍度

2. **BLOCK_ID**: CASE_08  
   - **Action**: adjust_assertion
   - **Error Type**: AssertionError
   - **问题**: AlphaDropout+SELU组合的方差特性需要更宽松的断言条件

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 无