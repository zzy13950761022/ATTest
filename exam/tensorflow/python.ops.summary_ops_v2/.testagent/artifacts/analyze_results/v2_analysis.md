# 测试执行分析报告

## 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 0  
- **错误**: 5
- **收集错误**: 否

## 待修复 BLOCK 列表 (1个)

### 1. HEADER (公共依赖)
- **Action**: fix_dependency
- **Error Type**: AttributeError
- **问题**: mock路径'tensorflow.python.ops.summary_ops_v2._summary_state'不可访问，需要修复fixture中的导入路径
- **影响范围**: 所有测试用例的setup阶段

## 延迟处理
- 4个测试用例因错误类型重复被标记为deferred
- 将在HEADER修复后重新评估

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无