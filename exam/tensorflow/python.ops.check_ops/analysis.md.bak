# 测试结果分析

## 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 9
- **错误**: 0
- **收集错误**: 否

## 待修复 BLOCK 列表 (3个)

### 1. CASE_01 - assert_equal_eager_mode
- **Action**: rewrite_block
- **Error Type**: AssertionError
- **原因**: mock_executing_eagerly被调用5次而非1次，需要修复mock设置

### 2. CASE_02 - assert_equal_graph_mode  
- **Action**: rewrite_block
- **Error Type**: AttributeError
- **原因**: 实际执行eager模式而非graph模式，mock未正确生效

### 3. CASE_03 - assert_less_static_failure
- **Action**: adjust_assertion
- **Error Type**: AssertionError
- **原因**: 错误消息不匹配预期，需要调整断言或修复mock

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无