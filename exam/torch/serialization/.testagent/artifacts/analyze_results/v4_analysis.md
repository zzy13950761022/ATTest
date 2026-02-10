# 测试结果分析

## 状态与统计
- **状态**: 未完全通过
- **通过**: 14个测试
- **失败**: 2个测试
- **错误**: 0个
- **集合错误**: 无

## 待修复BLOCK列表（2个）

### 1. CASE_09 - test_corrupted_file
- **错误类型**: AssertionError
- **修复动作**: adjust_assertion
- **原因**: 断言条件不匹配实际错误消息。实际错误消息为'pickle data was truncated'，但断言期望包含'unpickle'或'corrupt'

### 2. CASE_04 - test_weights_only_safety_mode[unsafe_object-True-False-None]
- **错误类型**: AttributeError
- **修复动作**: rewrite_block
- **原因**: lambda函数无法被pickle，需要在保存前跳过或使用可pickle的不安全对象

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无