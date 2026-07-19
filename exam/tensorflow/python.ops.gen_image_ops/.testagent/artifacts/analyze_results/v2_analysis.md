# 测试结果分析

## 状态与统计
- **状态**: 未完全通过
- **通过**: 4 个测试
- **失败**: 6 个测试
- **错误**: 0 个
- **集合错误**: 否

## 待修复 BLOCK 列表 (≤3)

### 1. CASE_02 - resize_bilinear 尺寸调整
- **错误类型**: InvalidArgumentError
- **Action**: rewrite_block
- **问题**: half_pixel_centers=True时align_corners必须为False，参数组合无效

### 2. CASE_03 - non_max_suppression 边界框抑制  
- **错误类型**: AssertionError
- **Action**: adjust_assertion
- **问题**: non_max_suppression返回的索引不保证排序，需要调整断言

### 3. CASE_05 - crop_and_resize 裁剪调整
- **错误类型**: AttributeError
- **Action**: rewrite_block
- **问题**: tf.reduce_std不存在，应使用tf.math.reduce_std

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无