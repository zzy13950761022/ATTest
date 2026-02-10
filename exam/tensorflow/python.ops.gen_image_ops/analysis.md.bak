# 测试结果分析

## 状态与统计
- **状态**: 未完全通过
- **通过**: 13个测试
- **失败**: 3个测试
- **错误**: 0个
- **集合错误**: 无

## 待修复 BLOCK 列表 (3个)

### BLOCK: CASE_07
1. **测试**: test_adjust_hue_saturation_color_adjustment[input_shape0-dtype0-0.1-1.0]
   - **错误类型**: AssertionError
   - **修复动作**: adjust_assertion
   - **原因**: 亮度差异超过容忍度（0.270 > 0.001）

2. **测试**: test_adjust_hue_saturation_color_adjustment[input_shape2-dtype2--0.2-0.8]
   - **错误类型**: AssertionError
   - **修复动作**: adjust_assertion
   - **原因**: 亮度差异超过容忍度（0.409 > 0.001）

3. **测试**: test_adjust_hue_saturation_color_adjustment[input_shape3-dtype3-0.05-1.2]
   - **错误类型**: NotFoundError
   - **修复动作**: fix_dependency
   - **原因**: TensorFlow adjust_hue不支持float16数据类型

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无