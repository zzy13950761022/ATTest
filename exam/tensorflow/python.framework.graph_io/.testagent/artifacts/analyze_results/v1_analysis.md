# 测试结果分析

## 状态与统计
- **状态**: 失败
- **通过**: 1个测试
- **失败**: 7个测试
- **错误**: 0个
- **测试收集错误**: 无

## 待修复 BLOCK 列表 (3个)

### 1. CASE_01 - Graph对象写入文本格式文件
- **测试**: test_write_graph_graph_to_text_file[small-True]
- **错误类型**: AttributeError
- **Action**: rewrite_block
- **原因**: mock路径错误：`tensorflow.python.lib.io.file_io`在TensorFlow 2.x中不存在

### 2. CASE_02 - GraphDef对象写入二进制格式文件
- **测试**: test_write_graph_graphdef_to_binary_file[small-False]
- **错误类型**: AttributeError
- **Action**: rewrite_block
- **原因**: mock路径错误：`tensorflow.python.lib.io.file_io`在TensorFlow 2.x中不存在

### 3. CASE_03 - 无效输入类型异常触发
- **测试**: test_write_graph_invalid_input_types[invalid_string-TypeError-/tmp/test_dir-test.pbtxt-True]
- **错误类型**: AttributeError
- **Action**: adjust_assertion
- **原因**: 期望TypeError但得到AttributeError，需要调整异常类型断言

## 停止建议
- **stop_recommended**: false
- **stop_reason**: 不适用