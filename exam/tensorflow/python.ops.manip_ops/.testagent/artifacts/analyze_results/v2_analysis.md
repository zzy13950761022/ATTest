# 测试执行分析报告

## 状态与统计
- **状态**: 失败（收集错误）
- **通过**: 0
- **失败**: 0
- **错误**: 1
- **收集错误**: 是

## 待修复 BLOCK 列表（1个）

### 1. HEADER 块
- **测试**: tests/test_tensorflow_python_ops_manip_ops.py::TestTensorFlowRoll
- **错误类型**: CollectionError
- **修复动作**: rewrite_block
- **原因**: 测试函数错误地定义为类方法，缺少self参数，应改为模块级函数

## 延迟处理
- test_multi_axis_roll: 错误类型重复，跳过该块
- test_same_axis_cumulative_roll: 错误类型重复，跳过该块
- test_boundary_conditions_empty_scalar: 错误类型重复，跳过该块
- test_type_validation_error_handling: 错误类型重复，跳过该块

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无