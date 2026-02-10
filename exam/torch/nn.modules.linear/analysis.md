# 测试结果分析

## 状态与统计
- **状态**: 未完全通过
- **通过**: 18 个测试
- **失败**: 2 个测试
- **错误**: 0 个
- **跳过**: 1 个
- **覆盖率**: 89%

## 待修复 BLOCK 列表 (≤3)

### 1. CASE_03 - Linear 不同数据类型
- **测试**: test_linear_different_dtypes
- **错误类型**: AssertionError
- **Action**: rewrite_block
- **问题**: state_dict加载后输出不匹配 - max_diff=1.97e+00, max_relative_diff=3.31e+02，需要修复state_dict保存/加载逻辑

### 2. CASE_09 - Linear 异常输入
- **测试**: test_linear_invalid_inputs
- **错误类型**: AssertionError
- **Action**: rewrite_block
- **问题**: 无效构造函数参数未抛出ValueError - Linear(in_features=0)应该抛出异常但未抛出

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无