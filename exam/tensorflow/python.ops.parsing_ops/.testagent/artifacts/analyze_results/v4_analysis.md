# 测试结果分析

## 状态与统计
- **状态**: 失败
- **通过**: 1
- **失败**: 0  
- **错误**: 21
- **跳过**: 1
- **覆盖率**: 12%

## 待修复 BLOCK 列表 (≤3)

### 1. HEADER - 公共依赖修复
- **Action**: fix_dependency
- **Error Type**: AttributeError
- **问题**: mock路径错误：`tensorflow.python.ops.parsing_ops.gen_parsing_ops`不存在
- **影响**: 所有测试用例的fixture初始化失败

### 2. HEADER - 公共依赖修复  
- **Action**: fix_dependency
- **Error Type**: AttributeError
- **问题**: mock路径错误：`tensorflow.python.ops.parsing_ops.gen_parsing_ops`不存在
- **影响**: CSV解析测试的fixture初始化失败

### 3. HEADER - 公共依赖修复
- **Action**: fix_dependency
- **Error Type**: AttributeError
- **问题**: mock路径错误：`tensorflow.python.ops.parsing_ops.gen_parsing_ops`不存在
- **影响**: 原始字节解码测试的fixture初始化失败

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 所有错误都是相同的AttributeError，需要修复公共mock路径问题