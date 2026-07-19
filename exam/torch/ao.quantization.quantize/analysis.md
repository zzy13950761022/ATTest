## 测试结果分析

### 状态与统计
- **状态**: 失败
- **通过**: 1个测试
- **失败**: 6个测试
- **错误**: 0个
- **收集错误**: 无

### 待修复BLOCK列表 (≤3)

1. **BLOCK: CASE_01** (TC-01: 基本浮点模型量化)
   - **Action**: rewrite_block
   - **Error Type**: AssertionError
   - **问题**: mock_prepare未被调用，可能导入路径或函数实现问题

2. **BLOCK: CASE_02** (TC-02: 原地量化验证)
   - **Action**: rewrite_block
   - **Error Type**: AssertionError
   - **问题**: mock_prepare未被调用，与CASE_01相同问题

3. **BLOCK: CASE_03** (TC-03: 自定义映射参数验证)
   - **Action**: rewrite_block
   - **Error Type**: AssertionError
   - **问题**: mock_convert未被调用，自定义映射测试

### 延迟处理
- CASE_04, CASE_05, G2的CASE_03: 错误类型重复，跳过该块

### 停止建议
- **stop_recommended**: false
- **原因**: 需要修复核心的mock调用问题