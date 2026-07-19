# tensorflow.python.ops.gen_decode_proto_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证 decode_proto_v2 函数正确解码序列化 Protocol Buffers 消息，将指定字段转换为对应 TensorFlow 数据类型
- 不在范围内的内容：Protocol Buffers 消息的序列化/反序列化过程、自定义 proto 消息定义验证、非标准 proto 特性支持

## 2. 输入与约束
- 参数列表：
  - bytes: Tensor/string 类型，形状为 batch_shape
  - message_type: string 类型，必需
  - field_names: list[string] 类型，必需
  - output_types: list[tf.DTypes] 类型，必需
  - descriptor_source: string 类型，默认 "local://"
  - message_format: string 类型，默认 "binary"（可选 "text"）
  - sanitize: bool 类型，默认 False
  - name: string 类型，可选
- 有效取值范围/维度/设备要求：
  - bytes 必须为有效的序列化 proto 数据
  - field_names 和 output_types 长度必须匹配
  - descriptor_source 支持 "local://" 或有效文件路径
  - 支持 CPU 和 GPU 设备
- 必需与可选组合：
  - bytes, message_type, field_names, output_types 为必需参数
  - descriptor_source, message_format, sanitize, name 为可选参数
- 随机性/全局状态要求：无随机性要求，操作为确定性

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 返回 _DecodeProtoV2Output 命名元组 (sizes, values)
  - sizes: int32 张量，形状为 (batch_size, num_fields)
  - values: output_types 类型的张量列表
- 容差/误差界（如浮点）：
  - 浮点类型允许机器精度误差
  - 整数类型必须精确匹配
- 状态变化或副作用检查点：
  - 无全局状态变化
  - 输出张量为密集张量，包含填充以容纳最大重复元素数

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - bytes 为空或无效 proto 数据
  - field_names 与 output_types 长度不匹配
  - 无效的 message_type 名称
  - 不支持的 output_types 类型
  - 无效的 descriptor_source 格式
  - 不支持的 message_format 值
- 边界值（空、None、0 长度、极端形状/数值）：
  - 空 field_names 列表
  - 空 bytes 张量
  - 超大 batch_size（内存边界）
  - 包含嵌套消息的字段
  - 重复字段和可选字段混合

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - Protocol Buffers 运行时库
  - 当 descriptor_source 为文件路径时，需要访问文件系统
  - TensorFlow 运行时环境
- 需要 mock/monkeypatch 的部分：
  - `tensorflow.python.framework.ops.convert_to_tensor`（输入验证）
  - `tensorflow.python.ops.gen_decode_proto_ops._op_def_library._apply_op_helper`（操作应用）
  - `tensorflow.python.eager.context.executing_eagerly`（执行模式检测）
  - `tensorflow.python.framework.dtypes.as_dtype`（类型转换）
  - 文件系统操作（当测试 descriptor_source 为文件路径时）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基本功能：有效 proto 数据解码为正确类型
  2. 参数验证：field_names 与 output_types 长度不匹配触发异常
  3. 格式支持：二进制和文本格式的 proto 解码
  4. 边界情况：空字段列表和空输入张量处理
  5. 数据类型：支持的所有 TensorFlow 数据类型转换
- 可选路径（中/低优先级合并为一组列表）：
  - 嵌套消息字段解码为 DT_STRING
  - map 字段作为 repeated 字段处理
  - enum 字段作为 int32 读取
  - uint64 类型表示为 DT_INT64
  - 不同 descriptor_source 值的行为
  - sanitize 参数的具体清理效果
  - 超大 batch_size 的性能和内存使用
  - 无效 proto 数据的错误恢复
- 已知风险/缺失信息（仅列条目，不展开）：
  - sanitize 参数的具体清理行为未明确说明
  - 子消息字段只能转换为 DT_STRING 的限制
  - 零尺寸维度防止机制的具体实现
  - 文件路径 descriptor_source 的错误处理细节
  - 并发调用时的线程安全性