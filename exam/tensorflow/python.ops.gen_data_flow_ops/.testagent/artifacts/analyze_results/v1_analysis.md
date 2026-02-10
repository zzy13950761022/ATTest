# 测试执行分析报告

## 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 0
- **错误**: 5
- **收集错误**: 否

## 待修复 BLOCK 列表 (≤3)

### 1. HEADER 块
- **测试**: test_fifo_queue_creation_enqueue_dequeue_full_flow
- **错误类型**: AttributeError
- **修复动作**: fix_dependency
- **原因**: mock路径错误：tensorflow.python.framework.ops.get_default_graph不存在

### 2. HEADER 块
- **测试**: test_tensor_array_dynamic_read_write_shape_preservation
- **错误类型**: AttributeError
- **修复动作**: fix_dependency
- **原因**: mock路径错误：tensorflow.python.framework.ops.get_default_graph不存在

### 3. HEADER 块
- **测试**: test_dynamic_partition_stitch_inverse_operation_verification
- **错误类型**: AttributeError
- **修复动作**: fix_dependency
- **原因**: mock路径错误：tensorflow.python.framework.ops.get_default_graph不存在

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无