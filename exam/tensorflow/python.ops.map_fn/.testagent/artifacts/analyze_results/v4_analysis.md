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
- **错误类型**: ValueError
- **修复动作**: rewrite_block
- **原因**: RaggedTensor处理形状不匹配：期望形状(2,)但实际形状(1,)，需要修复fn函数或调整map_fn调用

## 停止建议
- **stop_recommended**: false
- **原因**: 仅有一个测试失败，需要修复RaggedTensor处理逻辑