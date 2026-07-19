# 测试执行分析报告

## 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 0
- **错误**: 5
- **收集错误**: 否

## 待修复 BLOCK 列表 (≤3)

### 1. HEADER - fix_dependency
- **错误类型**: AttributeError
- **问题**: mock路径错误：tensorflow.python.eager.context.executing_eagerly不存在
- **影响测试**: 
  - test_fifo_queue_creation_enqueue_dequeue_full_flow
  - test_tensor_array_dynamic_read_write_shape_preservation
  - test_dynamic_partition_stitch_inverse_operation_verification

### 2. HEADER - fix_dependency
- **错误类型**: AttributeError
- **问题**: mock路径错误：tensorflow.python.eager.context.executing_eagerly不存在
- **影响测试**: test_tensor_array_dynamic_read_write_shape_preservation

### 3. HEADER - fix_dependency
- **错误类型**: AttributeError
- **问题**: mock路径错误：tensorflow.python.eager.context.executing_eagerly不存在
- **影响测试**: test_dynamic_partition_stitch_inverse_operation_verification

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无