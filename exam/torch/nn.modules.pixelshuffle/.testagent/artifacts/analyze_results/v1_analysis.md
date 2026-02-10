# 测试执行分析报告

## 状态与统计
- **状态**: 失败
- **通过**: 0
- **失败**: 0
- **错误**: 1
- **集合错误**: 是

## 待修复 BLOCK 列表 (1个)

### 1. HEADER - 文件依赖修复
- **错误类型**: FileNotFoundError
- **修复动作**: fix_dependency
- **问题描述**: 测试文件路径配置错误。测试计划期望分组文件 `test_torch_nn_modules_pixelshuffle_g1.py`，但实际文件名为 `test_torch_nn_modules_pixelshuffle.py`。

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无