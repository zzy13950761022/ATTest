# 测试分析报告

## 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 0
- **错误**: 1
- **集合错误**: 是

## 待修复 BLOCK 列表 (2个)

### 1. HEADER (测试文件组织)
- **测试**: test_torch_nn_modules_adaptive_g1.py
- **错误类型**: FileNotFoundError
- **修复动作**: rewrite_block
- **原因**: 需要按照test_plan.json的分组要求创建G1组测试文件

### 2. HEADER (测试文件组织)
- **测试**: test_torch_nn_modules_adaptive_g2.py
- **错误类型**: FileNotFoundError
- **修复动作**: rewrite_block
- **原因**: 需要按照test_plan.json的分组要求创建G2组测试文件

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无