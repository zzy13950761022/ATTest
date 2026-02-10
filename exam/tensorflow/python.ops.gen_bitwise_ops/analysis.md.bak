# 测试分析报告

## 状态与统计
- **状态**: 未完全通过
- **通过**: 20
- **失败**: 1
- **错误**: 0
- **跳过**: 1

## 待修复 BLOCK 列表
1. **BLOCK_ID**: CASE_04
   - **Action**: adjust_assertion
   - **Error Type**: AssertionError
   - **原因**: TensorFlow left_shift在移位>=位宽时未返回0，而是返回非零值(2147483648)，需要调整断言逻辑

## 停止建议
`stop_recommended`: true
`stop_reason`: 与上一轮失败集合完全重复：相同的测试用例(test_left_shift_boundary)和相同的错误类型(AssertionError)，表明修复尝试未解决问题或问题为TensorFlow实现特性