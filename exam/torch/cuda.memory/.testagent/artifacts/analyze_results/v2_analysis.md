# 测试分析报告

## 状态与统计
- **状态**: 未完全通过
- **通过测试**: 0
- **失败测试**: 0  
- **错误测试**: 0
- **跳过测试**: 8
- **覆盖率**: 12%

## 待修复 BLOCK 列表（本轮最多3个）

### 1. HEADER - CUDA可用性检查
- **Action**: rewrite_block
- **Error Type**: SkipTest
- **原因**: HEADER中的CUDA可用性检查导致所有测试被跳过，需要修改检查逻辑或添加mock支持

### 2. CASE_05 - 内存统计字典结构完整性测试
- **Action**: adjust_assertion  
- **Error Type**: SkipTest
- **原因**: 测试因CUDA不可用被跳过，需要调整测试逻辑以在没有CUDA的环境中运行

### 3. CASE_06 - memory_summary格式化输出测试
- **Action**: adjust_assertion
- **Error Type**: SkipTest
- **原因**: 测试因CUDA不可用被跳过，需要调整测试逻辑以在没有CUDA的环境中运行

## 延迟修复
- `test_memory_stats_function_signatures`: 依赖HEADER修复，错误类型重复
- `test_module_import`: 依赖HEADER修复，错误类型重复

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无