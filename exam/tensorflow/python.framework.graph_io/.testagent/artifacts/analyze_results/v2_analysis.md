## 测试结果分析

### 状态与统计
- **状态**: 失败
- **通过**: 3个测试
- **失败**: 5个测试
- **错误**: 0个

### 待修复 BLOCK 列表（本轮修复 3 个）

1. **BLOCK_ID**: CASE_01
   - **Action**: rewrite_block
   - **Error Type**: AttributeError
   - **原因**: mock路径错误：tensorflow.python.lib.io.file_io不存在

2. **BLOCK_ID**: CASE_02
   - **Action**: rewrite_block
   - **Error Type**: AttributeError
   - **原因**: mock路径错误：tensorflow.python.lib.io.file_io不存在

3. **BLOCK_ID**: CASE_04
   - **Action**: rewrite_block
   - **Error Type**: AttributeError
   - **原因**: mock路径错误：tensorflow.python.lib.io.file_io不存在

### 延迟处理
- test_write_graph_default_as_text (FOOTER): 错误类型重复，跳过该块
- test_write_graph_gcs_path_no_directory_creation (FOOTER): 错误类型重复，跳过该块

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 需要修复mock路径问题