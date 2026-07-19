## 测试结果分析

### 状态与统计
- **状态**: 未完全通过
- **通过**: 1个测试
- **失败**: 4个测试
- **错误**: 0个
- **集合错误**: 无

### 待修复 BLOCK 列表 (2个)

1. **BLOCK_ID**: CASE_01
   - **Action**: rewrite_block
   - **Error Type**: AssertionError
   - **原因**: SequenceCategoricalColumn是包装类，key属性在categorical_column内部对象上，需要修改测试逻辑

2. **BLOCK_ID**: CASE_01  
   - **Action**: adjust_assertion
   - **Error Type**: ValueError
   - **原因**: 参数扩展中的default_value=-1无效，应修改为有效值或标记为xfail

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 无