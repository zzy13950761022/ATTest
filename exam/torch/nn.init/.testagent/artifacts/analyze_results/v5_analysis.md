# 测试结果分析

## 状态与统计
- **状态**: 未完全通过
- **通过**: 19 个测试
- **失败**: 1 个测试
- **错误**: 0 个
- **集合错误**: 无

## 待修复 BLOCK 列表 (1个)

### 1. FOOTER 块
- **测试**: `tests/test_torch_nn_init_g2.py::test_invalid_inputs`
- **错误类型**: AssertionError
- **修复动作**: adjust_assertion
- **原因**: 错误消息正则表达式不匹配。期望匹配 `'nonlinearity not found'`，但实际错误消息为 `'Unsupported nonlinearity invalid_nonlinearity'`

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 无