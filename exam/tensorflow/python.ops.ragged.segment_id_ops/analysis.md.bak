# 测试执行分析报告

## 状态与统计
- **状态**: 未完全通过
- **通过测试**: 25
- **失败测试**: 4
- **错误测试**: 0
- **测试收集错误**: 否

## 待修复 BLOCK 列表 (≤3)

### 1. HEADER - 辅助函数修复
- **测试**: test_helper_functions
- **错误类型**: TypeError
- **修复动作**: rewrite_block
- **问题描述**: 辅助函数assert_tensor_equal在处理None类型错误消息时出现TypeError

### 2. CASE_04 - int64支持测试调整
- **测试**: test_int64_support[splits4-dtype4-None-None]
- **错误类型**: AssertionError
- **修复动作**: adjust_assertion
- **问题描述**: 零长度段处理时roundtrip验证失败，形状不匹配(3,) vs (4,)

### 3. CASE_05 - 无效输入测试调整
- **测试**: test_invalid_inputs
- **错误类型**: AssertionError
- **修复动作**: adjust_assertion
- **问题描述**: splits不以0开始时未抛出InvalidArgumentError，需要调整断言或实现

## 延迟处理
- **test_num_segments_parameter**: 错误类型重复，跳过该块

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无