# 测试分析报告

## 状态与统计
- **状态**: 未完全通过
- **通过**: 13
- **失败**: 1
- **错误**: 0
- **收集错误**: 否

## 待修复 BLOCK 列表 (1个)

### 1. HEADER
- **测试**: `TestVerifyTensorAllFiniteV1.test_v1_calls_v2`
- **错误类型**: `AttributeError`
- **修复动作**: `rewrite_block`
- **原因**: mock路径错误：tensorflow.python在mock时不可访问，需要修复mock路径

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无