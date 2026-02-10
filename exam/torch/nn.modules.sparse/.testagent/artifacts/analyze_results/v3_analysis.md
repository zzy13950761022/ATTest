# 测试结果分析

## 状态与统计
- **状态**: 未完全通过
- **通过**: 9个测试
- **失败**: 5个测试
- **错误**: 0个
- **覆盖率**: 69%

## 待修复 BLOCK 列表（本轮修复 ≤3）

### 1. BLOCK: HEADER
- **测试**: test_embedding_invalid_parameters
- **错误类型**: Failed (DID NOT RAISE ValueError)
- **修复动作**: rewrite_block
- **原因**: Embedding构造函数未对num_embeddings<=0抛出ValueError，需要修复参数验证逻辑

### 2. BLOCK: CASE_02
- **测试**: test_embedding_edge_cases
- **错误类型**: IndexError
- **修复动作**: rewrite_block
- **原因**: 负padding_idx(-1)处理逻辑错误，需要修复负索引转换逻辑

### 3. BLOCK: CASE_03
- **测试**: test_embeddingbag_edge_cases
- **错误类型**: RuntimeError
- **修复动作**: rewrite_block
- **原因**: 负padding_idx(-1)导致索引越界，需要修复负索引处理逻辑

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无