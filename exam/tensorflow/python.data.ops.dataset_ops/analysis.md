## 测试结果分析

### 状态统计
- **状态**: 未完全通过
- **通过**: 11个测试
- **失败**: 2个测试
- **错误**: 0个
- **收集错误**: 无

### 待修复BLOCK列表 (<=3)

1. **BLOCK_ID**: CASE_03
   - **测试**: TestDatasetOps::test_map_operation[tensor-data_shape0-float32-cpu-None-False-lambda x: x * 2]
   - **错误类型**: ValueError
   - **修复动作**: rewrite_block
   - **原因**: error_func在map创建时执行导致ValueError，需要修复TensorFlow图编译问题

2. **BLOCK_ID**: CASE_03
   - **测试**: TestDatasetOps::test_map_operation[tensor-data_shape1-float32-cpu-None-False-lambda x: tf.reduce_sum(x)]
   - **错误类型**: ValueError
   - **修复动作**: rewrite_block
   - **原因**: 与第一个失败相同错误类型，同一BLOCK_ID

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 无