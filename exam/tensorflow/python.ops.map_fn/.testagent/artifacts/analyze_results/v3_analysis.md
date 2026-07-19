# 测试结果分析

## 状态与统计
- **状态**: 未完全通过
- **通过**: 12 个测试
- **失败**: 1 个测试
- **错误**: 0 个
- **集合错误**: 否

## 待修复 BLOCK 列表 (1个)

### 1. CASE_03 - RaggedTensor输入处理
- **测试**: `test_ragged_tensor_input[2-fn_output_signature1-10-True-ragged_test]`
- **错误类型**: InvalidArgumentError
- **修复动作**: rewrite_block
- **原因**: RaggedTensor测试中fn_output_signature指定为float32但函数返回int32，导致TensorArray类型不匹配。需要确保函数返回值类型与fn_output_signature一致。

## 停止建议
- **stop_recommended**: false
- **原因**: 新的失败类型，需要修复类型匹配问题