## 测试结果分析

### 状态与统计
- **状态**: 未完全通过
- **通过**: 13个测试
- **失败**: 3个测试
- **错误**: 0个
- **收集错误**: 无

### 待修复 BLOCK 列表（本轮最多3个）

1. **BLOCK: CASE_02** - categorical_column_with_vocabulary_list基础创建
   - **Action**: rewrite_block
   - **Error Type**: ValueError
   - **原因**: TensorFlow实现检查`default_value != -1`就报错，即使`default_value=None`。测试需要调整以匹配实际行为。

2. **BLOCK: CASE_04** - embedding_column维度验证
   - **Action**: adjust_assertion  
   - **Error Type**: AssertionError
   - **原因**: embedding_column未对无效combiner值`"invalid_combiner"`引发ValueError，需要调整测试或检查实际验证逻辑。

3. **BLOCK: CASE_04** - embedding_column维度验证（第二个参数化测试）
   - **Action**: adjust_assertion
   - **Error Type**: AssertionError
   - **原因**: 与上一个失败相同，都是无效combiner验证问题。

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 无