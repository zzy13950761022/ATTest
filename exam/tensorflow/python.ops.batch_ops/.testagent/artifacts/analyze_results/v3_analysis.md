## 测试结果分析

### 状态与统计
- **状态**: 未完全通过
- **通过**: 0
- **失败**: 2
- **错误**: 5
- **跳过**: 1

### 待修复 BLOCK 列表 (2个)

1. **BLOCK_ID**: HEADER
   - **Action**: rewrite_block
   - **Error Type**: AttributeError
   - **Note**: mock_validate_allowed_batch_sizes试图mock不存在的函数

2. **BLOCK_ID**: CASE_01
   - **Action**: rewrite_block
   - **Error Type**: TypeError
   - **Note**: mock_defun不接受autograph参数，需要修复fixture

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 需要修复HEADER block中的fixture问题