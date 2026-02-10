# 测试结果分析

## 状态与统计
- **状态**: 未完全通过
- **通过**: 5个测试
- **失败**: 5个测试
- **错误**: 0个
- **集合错误**: 否

## 待修复 BLOCK 列表 (≤3)

### 1. BLOCK: CASE_01
- **测试**: test_basic_tensor_mapping[dtype1-shape1-5-False-True-False-extended_test]
- **错误类型**: AttributeError
- **修复动作**: adjust_assertion
- **原因**: Tensor.name属性在eager模式下不可用，需要移除或调整name检查断言

### 2. BLOCK: CASE_02
- **测试**: test_nested_tensor_input[dtype0-shape0-fn_output_signature0-10-test_map]
- **错误类型**: AttributeError
- **修复动作**: adjust_assertion
- **原因**: Tensor.name属性在eager模式下不可用，需要移除或调整name检查断言

### 3. BLOCK: CASE_05
- **测试**: test_fn_signature_requires_output_signature[input_dtype0-output_dtype0-fn_output_signature0]
- **错误类型**: InvalidArgumentError
- **修复动作**: rewrite_block
- **原因**: TensorArray dtype不匹配：期望float32但写入int32，需要修复类型转换逻辑

## 延迟处理
- test_parallel_iterations_zero_raises_error: 错误类型重复，跳过该块
- test_dtype_deprecated_warning: 错误类型重复，跳过该块

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无