# 测试结果分析

## 状态与统计
- **状态**: 未完全通过
- **通过**: 16个测试
- **失败**: 3个测试
- **错误**: 0个错误
- **跳过**: 2个测试

## 待修复BLOCK列表（≤3个）

### 1. HEADER - 核心修复
- **Action**: rewrite_block
- **Error Type**: RuntimeError
- **问题**: graph模式下EagerTensor无法在tf.compat.v1.Session中使用
- **影响**: 所有graph模式测试用例
- **解决方案**: 需要修复create_tensor函数，使其在graph模式下返回正确的Tensor类型

## 延迟处理
1. **test_bitwise_or_broadcast[int8-shape_x1-shape_y1-values_x1-values_y1-cpu-graph]** - 错误类型重复，跳过该块
2. **test_invert_signed_integer[uint32-shape1-values1-cpu-graph]** - 错误类型重复，跳过该块

## 停止建议
- **stop_recommended**: false
- **原因**: 需要修复graph模式下的EagerTensor问题，修复HEADER后可解决所有相关失败