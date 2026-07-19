# 测试结果分析

## 状态与统计
- **状态**: 失败
- **通过**: 1
- **失败**: 0  
- **错误**: 21
- **收集错误**: 否

## 待修复 BLOCK 列表 (1个)

### 1. HEADER - mock路径修复
- **错误类型**: AttributeError
- **Action**: fix_dependency
- **问题**: mock路径`'tensorflow.python.ops.gen_parsing_ops'`在TensorFlow 2.x中不存在，需要修正为正确的模块路径
- **影响范围**: 所有测试用例（21个错误）

## 延迟处理
- 其余20个测试用例因相同错误类型被标记为deferred
- 原因: "错误类型重复，跳过该块"

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 需要先修复HEADER中的mock路径问题，这是所有测试失败的根本原因