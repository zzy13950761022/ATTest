## 测试结果分析

### 状态与统计
- **状态**: 未完全通过
- **通过**: 10个测试
- **失败**: 3个测试
- **错误**: 0个
- **覆盖率**: 78%

### 待修复 BLOCK 列表 (3个)

1. **BLOCK_ID**: HEADER
   - **测试**: test_adaptive_softmax_forward_basic[20-200-cutoffs1-2.0-True-4-dtype1-cpu]
   - **错误类型**: RuntimeError
   - **Action**: fix_dependency
   - **原因**: create_adaptive_softmax函数未传递dtype参数，导致模型权重类型与输入类型不匹配

2. **BLOCK_ID**: FOOTER
   - **测试**: test_adaptive_softmax_shape_mismatch
   - **错误类型**: RuntimeError
   - **Action**: adjust_assertion
   - **原因**: 测试期望IndexError但实际抛出RuntimeError，需要调整断言类型

3. **BLOCK_ID**: FOOTER
   - **测试**: test_adaptive_softmax_invalid_parameters_g2
   - **错误类型**: ZeroDivisionError
   - **Action**: adjust_assertion
   - **原因**: div_value=0.0导致ZeroDivisionError而非ValueError，需要调整测试逻辑

### 停止建议
- **stop_recommended**: false
- **stop_reason**: 无