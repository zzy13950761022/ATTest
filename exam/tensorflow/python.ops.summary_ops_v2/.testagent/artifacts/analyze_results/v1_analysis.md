# 测试执行分析报告

## 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 0
- **错误**: 3
- **收集错误**: 否

## 待修复BLOCK列表 (1个)

### 1. HEADER (fixture修复)
- **Action**: fix_dependency
- **Error Type**: AttributeError
- **问题**: tensorflow模块没有python属性，需要修复mock路径
- **影响测试**: 所有测试用例的setup阶段

## 延迟处理
- 2个测试用例因错误类型重复被延迟处理

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无