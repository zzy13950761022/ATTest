# 测试执行分析报告

## 状态与统计
- **状态**: 未完全通过
- **通过**: 2 个测试用例
- **失败**: 4 个测试用例
- **错误**: 0 个测试用例
- **收集错误**: 无

## 待修复 BLOCK 列表 (≤3)

### 1. CASE_01 - test_basic_tf_module_save
- **错误类型**: AttributeError
- **修复动作**: rewrite_block
- **原因**: mock路径错误：tensorflow.python不是TensorFlow 2.x的公共API，应使用tf.saved_model.save_impl等公共API路径

### 2. CASE_02 - test_tf_module_with_tf_function
- **错误类型**: AttributeError
- **修复动作**: rewrite_block
- **原因**: mock路径错误：tensorflow.python不是公共API，需要修正为正确的模块路径

### 3. CASE_03 - test_explicit_signatures_parameter
- **错误类型**: AttributeError
- **修复动作**: rewrite_block
- **原因**: mock路径错误：tensorflow.python不是公共API，需要修正为正确的模块路径

## 延迟修复
- **test_trackable_object_with_variables (CASE_04)**: 错误类型重复，跳过该块

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无