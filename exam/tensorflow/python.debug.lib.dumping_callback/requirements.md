# tensorflow.python.debug.lib.dumping_callback 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证 `enable_dump_debug_info` 正确启用调试信息转储到指定目录
  - 验证 `disable_dump_debug_info` 正确禁用转储并清理资源
  - 测试不同 `tensor_debug_mode` 下张量信息提取的准确性
  - 验证过滤条件（op_regex, tensor_dtypes）的正确应用
  - 确保环形缓冲区功能按预期工作
- 不在范围内的内容
  - 调试信息文件格式的详细解析
  - 性能基准测试和内存使用量化
  - 跨平台文件系统兼容性测试
  - 第三方工具对转储文件的处理

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - dump_root: str, 必需，有效目录路径
  - tensor_debug_mode: str, 默认'NO_TENSOR', 枚举值：NO_TENSOR/CURT_HEALTH/CONCISE_HEALTH/FULL_HEALTH/SHAPE
  - circular_buffer_size: int, 默认1000, 支持<=0（禁用缓冲区）
  - op_regex: str/None, 可选，有效正则表达式
  - tensor_dtypes: list/tuple/callable/None, 可选，DType对象或字符串列表
- 有效取值范围/维度/设备要求
  - dump_root: 可写目录路径，支持绝对/相对路径
  - circular_buffer_size: 整数，支持正数、0、负数
  - tensor_dtypes: 仅浮点类型（float32, float64, bfloat16）支持健康检查模式
  - TPU环境需先调用 `tf.config.set_soft_device_placement(True)`
- 必需与可选组合
  - dump_root 为必需参数
  - 其他参数均为可选，有默认值
  - op_regex 和 tensor_dtypes 为逻辑与关系
- 随机性/全局状态要求
  - 使用线程局部存储管理状态
  - 多次调用相同dump_root需幂等
  - 全局回调注册/注销操作

## 3. 输出与判定
- 期望返回结构及关键字段
  - enable_dump_debug_info: 返回DebugEventsWriter实例，需验证flush方法可用
  - disable_dump_debug_info: 无返回值
- 容差/误差界（如浮点）
  - 浮点张量健康检查需准确检测inf/NaN
  - 形状信息提取需与张量实际形状一致
  - 元素计数统计需精确
- 状态变化或副作用检查点
  - 验证目标目录创建和文件写入
  - 验证全局回调正确注册和注销
  - 验证线程局部状态正确更新
  - 验证资源清理（文件句柄、内存）

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 无效dump_root路径（不可写、不存在父目录）
  - 无效tensor_debug_mode字符串
  - 无效circular_buffer_size类型
  - 无效op_regex正则表达式
  - 无效tensor_dtypes格式
- 边界值（空、None、0长度、极端形状/数值）
  - dump_root为空字符串
  - circular_buffer_size为0或负数
  - op_regex为空字符串
  - tensor_dtypes为空列表
  - 极端大张量形状测试
  - 包含inf/NaN的浮点张量

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - 文件系统：需要可写目录权限
  - 磁盘空间：需考虑调试信息文件大小
  - TPU设备：特殊配置要求
- 需要mock/monkeypatch的部分
  - 文件系统操作（os.makedirs, open等）
  - 全局回调注册机制
  - DebugEventsWriter实例方法
  - 线程局部存储访问

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. 基本启用/禁用流程验证
  2. 不同tensor_debug_mode的功能验证
  3. 过滤条件（op_regex, tensor_dtypes）正确性
  4. 环形缓冲区功能测试
  5. 异常参数处理和错误恢复
- 可选路径（中/低优先级合并为一组列表）
  - 并发调用和线程安全性
  - 多次启用相同dump_root的幂等性
  - 不同设备类型（CPU/GPU/TPU）兼容性
  - 大文件写入和磁盘空间处理
  - 长时间运行的内存泄漏检查
- 已知风险/缺失信息（仅列条目，不展开）
  - 缺少完整类型注解
  - 文件格式细节未明确
  - 性能影响未量化
  - 跨平台兼容性未说明
  - 资源清理的完整性