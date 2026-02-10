# 测试分析报告

## 状态与统计
- **状态**: 成功
- **通过**: 2个测试
- **失败**: 0个测试
- **错误**: 0个测试
- **跳过**: 6个测试
- **预期失败**: 1个测试
- **总覆盖率**: 75%

## 待修复 BLOCK 列表（≤3）

### 1. CASE_09 - emit_nvtx 基本功能测试
- **Action**: mark_xfail
- **Error Type**: XFailed
- **Note**: 测试环境没有CUDA支持，标记为预期失败

### 2. CASE_06 - 高级功能测试
- **Action**: add_case
- **Error Type**: MissingCoverage
- **Note**: 覆盖率报告显示高级功能测试缺失，需要实现CASE_06

### 3. HEADER - 测试fixture
- **Action**: adjust_assertion
- **Error Type**: MissingCoverage
- **Note**: HEADER块中的fixture函数有未覆盖的行，需要增强测试

## Stop Recommendation
- **stop_recommended**: false
- **stop_reason**: 测试运行成功，没有重复的失败模式