# 测试分析报告

## 状态与统计
- **状态**: 失败
- **通过**: 1个测试
- **失败**: 6个测试
- **错误**: 0个
- **收集错误**: 无

## 待修复 BLOCK 列表 (3个)

### 1. CASE_01 - 基本浮点模型量化验证
- **测试**: test_basic_float_model_quantization
- **错误类型**: AssertionError
- **Action**: rewrite_block
- **问题**: mock_prepare没有被调用，quantize函数可能没有按预期调用prepare

### 2. CASE_04 - 校准函数参数传递验证
- **测试**: test_calibration_function_parameter_passing
- **错误类型**: AssertionError
- **Action**: rewrite_block
- **问题**: run_fn没有被调用，需要检查quantize函数是否真的调用了run_fn

### 3. CASE_03 - 自定义映射参数验证
- **测试**: test_custom_mapping_parameter
- **错误类型**: AssertionError
- **Action**: rewrite_block
- **问题**: mock_convert没有被调用，需要检查custom mapping参数传递

## 延迟处理
- CASE_02 (test_inplace_quantization): 错误类型重复，跳过该块
- CASE_05 (test_different_model_architecture_compatibility): 错误类型重复，跳过该块
- G2 CASE_03 (test_custom_mapping_parameter): 错误类型重复，跳过该块

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无