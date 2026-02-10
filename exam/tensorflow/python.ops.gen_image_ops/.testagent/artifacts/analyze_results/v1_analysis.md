# 测试执行分析报告

## 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 10
- **错误**: 0
- **收集错误**: 否

## 待修复 BLOCK 列表 (≤3)

### 1. CASE_01 - decode_jpeg 基本解码
- **测试**: test_decode_jpeg_basic[3-True-False-1.0-]
- **错误类型**: TypeError
- **修复动作**: adjust_assertion
- **原因**: tf.math.reduce_std不能处理uint8类型，需要转换为float类型

### 2. CASE_02 - resize_bilinear 尺寸调整
- **测试**: test_resize_bilinear_size_adjustment[input_shape0-output_shape0-dtype0-False-False]
- **错误类型**: AttributeError
- **修复动作**: rewrite_block
- **原因**: dtype.numpy_dtype属性不存在，应使用dtype.as_numpy_dtype

### 3. CASE_03 - non_max_suppression 边界框抑制
- **测试**: test_non_max_suppression_bbox_suppression[boxes_shape0-scores_shape0-5-0.5-0.0-dtype0]
- **错误类型**: AttributeError
- **修复动作**: rewrite_block
- **原因**: dtype.numpy_dtype属性不存在，应使用dtype.as_numpy_dtype

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无