# tensorflow.python.ops.string_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试字符串张量操作模块的核心功能，包括正则匹配、替换、字符串分割、连接、长度计算、大小写转换等
- 不在范围内的内容：非字符串张量操作、底层 gen_string_ops 实现细节、re2 库内部实现

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - `input`: string 类型张量，任意形状
  - `pattern`: string 类型标量或张量，re2 正则语法
  - `rewrite`: string 类型，替换文本
  - `separator`: string 类型，默认空字符串
  - `unit`: string 枚举，'BYTE' 或 'UTF8_CHAR'，默认 'BYTE'
  - `skip_empty`: bool 类型，默认 True
  - `name`: string 类型，默认 None

- 有效取值范围/维度/设备要求：
  - 字符串张量支持任意形状
  - 正则模式支持标量或与输入匹配的形状
  - UTF8_CHAR 单位要求输入为有效 UTF-8 字符串
  - 支持 CPU 和 GPU 设备

- 必需与可选组合：
  - `regex_full_match`: input, pattern 必需
  - `regex_replace`: input, pattern, rewrite 必需，replace_global 可选
  - `string_length`: input 必需，unit 可选
  - `string_join`: inputs 必需，separator 可选
  - `string_split`: source 必需，sep/delimiter 可选

- 随机性/全局状态要求：无随机性，无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 正则匹配：bool 类型张量，保持输入形状
  - 字符串操作：string 类型张量，保持输入形状
  - 长度计算：int32 类型张量，保持输入形状
  - 分割操作：SparseTensor 类型，包含 indices/values/dense_shape

- 容差/误差界（如浮点）：字符串操作无浮点误差，精确匹配要求

- 状态变化或副作用检查点：无状态变化，无副作用

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非字符串类型输入触发 InvalidArgumentError
  - 无效 UTF-8 字符串使用 UTF8_CHAR 单位触发异常
  - 无效正则表达式语法触发 InvalidArgumentError
  - 形状不匹配触发 ValueError

- 边界值（空、None、0 长度、极端形状/数值）：
  - 空字符串输入
  - 空张量（shape=[0]）
  - 超大字符串长度
  - 特殊字符：Unicode 表情、控制字符
  - 多行字符串
  - 空分隔符
  - None 值处理

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：无外部依赖
- 需要 mock/monkeypatch 的部分：
  - `tensorflow.python.ops.gen_string_ops` 底层操作
  - `tensorflow.python.framework.ops.convert_to_tensor` 类型转换
  - `tensorflow.python.ops.array_ops.shape` 形状获取
  - `tensorflow.python.ops.control_flow_ops.assert_positive` 断言检查

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 正则匹配函数对 re2 语法的正确解析
  2. 字符串长度计算在 BYTE 和 UTF8_CHAR 单位的差异
  3. 字符串分割对空值和分隔符的处理
  4. 字符串连接保持形状一致性
  5. 无效 UTF-8 输入在 UTF8_CHAR 单位的异常处理

- 可选路径（中/低优先级合并为一组列表）：
  - 特殊 Unicode 字符处理
  - 超大字符串性能测试
  - 多设备（CPU/GPU）一致性
  - 形状广播行为
  - 正则替换的全局/非全局模式
  - 空张量和零维张量
  - 字符串大小写转换边界

- 已知风险/缺失信息（仅列条目，不展开）：
  - re2 与 Python re 语法差异
  - UTF8_CHAR 对无效 UTF-8 的具体行为
  - 多字节字符的长度计算精度
  - 正则性能差异（static vs 动态模式）
  - 稀疏张量分割的内存使用