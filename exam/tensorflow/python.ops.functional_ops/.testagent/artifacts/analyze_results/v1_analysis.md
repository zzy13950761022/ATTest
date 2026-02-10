# 测试结果分析

## 状态统计
- **状态**: 未完全通过
- **通过**: 3个测试
- **失败**: 3个测试
- **错误**: 0个
- **集合错误**: 无

## 待修复BLOCK列表（最多3个）

### 1. BLOCK: CASE_02
- **测试**: `test_scan_accumulation[dtype1-None-True-False-graph]`
- **错误类型**: AttributeError
- **修复动作**: rewrite_block
- **原因**: Tensor.graph属性在eager执行模式下不可用，需要修复图模式下的expected tensor处理

### 2. BLOCK: CASE_03  
- **测试**: `test_if_conditional_branch[True-input_shape0-dtype0-add_one-subtract_one-eager]`
- **错误类型**: AttributeError
- **修复动作**: fix_dependency
- **原因**: mock路径'tensorflow.python.ops.gen_functional_ops._if'不存在，需要修复导入路径

## 延迟处理
- `test_if_conditional_branch[False-input_shape1-dtype1-multiply_two-divide_two-graph]`: 错误类型重复，跳过该块

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无