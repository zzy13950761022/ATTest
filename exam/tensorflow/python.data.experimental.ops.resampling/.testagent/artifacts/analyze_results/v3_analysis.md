## 测试结果分析

### 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 4
- **错误**: 0
- **测试收集错误**: 否

### 待修复 BLOCK 列表 (3个)

1. **CASE_01** - TestRejectionResample.test_basic_functionality
   - **Action**: rewrite_block
   - **Error Type**: AssertionError
   - **Note**: mock配置错误，result不是预期的transformed_dataset

2. **CASE_02** - TestRejectionResample.test_deprecation_warning
   - **Action**: rewrite_block
   - **Error Type**: AssertionError
   - **Note**: 弃用警告未正确捕获，mock路径可能不正确

3. **CASE_03** - TestRejectionResample.test_distribution_adjustment
   - **Action**: rewrite_block
   - **Error Type**: AssertionError
   - **Note**: mock_rejection_resample未被调用，需要检查mock配置

### 延迟处理
- CASE_04: 错误类型重复，跳过该块

### 停止建议
- **stop_recommended**: false