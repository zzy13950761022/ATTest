# 测试执行分析报告

## 状态与统计
- **状态**: 未完全通过
- **通过**: 5个测试用例
- **失败**: 2个测试用例
- **错误**: 0个
- **集合错误**: 无

## 待修复BLOCK列表（≤3个）

### 1. CASE_02 - parse_example稀疏稠密特征混合解析
- **Action**: rewrite_block
- **Error Type**: TypeError
- **问题**: parse_example()函数调用缺少必需的'names'参数
- **影响测试**: test_parse_example_mixed_sparse_dense_features

## 延迟处理
- 第二个失败用例（空特征Example解析）与第一个失败用例错误类型相同，已标记为deferred

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无