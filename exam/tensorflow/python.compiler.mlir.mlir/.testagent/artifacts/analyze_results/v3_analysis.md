## 测试结果分析

### 状态与统计
- **状态**: 失败
- **通过**: 3个测试
- **失败**: 0个测试  
- **错误**: 3个测试
- **收集错误**: 无

### 待修复 BLOCK 列表 (1个)

1. **BLOCK_ID**: HEADER
   - **Action**: rewrite_block
   - **Error Type**: AttributeError
   - **问题**: `text_proto_graph_def` fixture 中使用 `tf.compat.v1.text_format.Parse()` 失败，当前 TensorFlow 版本中该属性不存在

### 延迟处理
- 2个测试因错误类型重复被标记为 deferred
- 所有错误都源于同一个 HEADER block 中的 fixture 问题

### 停止建议
- **stop_recommended**: false
- 需要修复 HEADER block 中的 fixture 问题