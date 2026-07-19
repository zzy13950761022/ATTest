# tensorflow.python.data.experimental.ops.readers 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试CSV、SQL、TFRecord数据集的读取、解析、批处理功能；验证类型推断、文件模式匹配、并行处理逻辑
- 不在范围内的内容：底层TensorFlow核心API、第三方数据库驱动实现、文件系统操作的具体实现

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - make_csv_dataset: file_pattern(str/list), batch_size(int), column_names(list/None), column_defaults(list/None), label_name(str/None), select_columns(list/None)
  - CsvDataset: file_pattern(str/list), record_defaults(list), select_cols(list/None)
  - SqlDataset: driver_name(str仅'sqlite'), data_source_name(str), query(str), output_types(tuple)
  - make_batched_features_dataset: file_pattern(str/list), batch_size(int), features(dict)
  - make_tf_record_dataset: file_pattern(str/list), batch_size(int), compression_type(str/None)

- 有效取值范围/维度/设备要求：
  - batch_size > 0
  - file_pattern支持通配符(*, ?)和列表
  - 数据类型仅限int32、int64、float32、float64、string
  - 支持GZIP、ZLIB压缩格式
  - SQL仅支持sqlite驱动

- 必需与可选组合：
  - make_csv_dataset: file_pattern必需，其他可选
  - CsvDataset: file_pattern和record_defaults必需
  - SqlDataset: 所有参数必需

- 随机性/全局状态要求：
  - 文件读取顺序可能随机（取决于文件系统）
  - 批处理顺序可配置为随机或顺序

## 3. 输出与判定
- 期望返回结构及关键字段：
  - make_csv_dataset: (features, labels)元组的Dataset
  - CsvDataset: 记录元组的Dataset
  - SqlDataset: 查询结果元组的Dataset
  - make_batched_features_dataset: 特征字典的Dataset
  - make_tf_record_dataset: TFRecord记录的Dataset

- 容差/误差界（如浮点）：
  - 浮点数解析精度与Python float一致
  - 字符串编码使用UTF-8
  - 数值类型转换遵循TensorFlow类型转换规则

- 状态变化或副作用检查点：
  - 文件句柄正确关闭
  - 数据库连接正确释放
  - 内存使用量在合理范围内
  - 无残留的临时文件或资源

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 无效文件路径：NotFoundError
  - 无效batch_size：InvalidArgumentError
  - 不支持的压缩类型：InvalidArgumentError
  - 类型不匹配：ValueError
  - 无效SQL驱动：ValueError
  - 查询语法错误：SQL相关异常

- 边界值（空、None、0长度、极端形状/数值）：
  - 空文件或空查询结果
  - None作为参数值
  - 0长度列表作为column_names或select_columns
  - 极大batch_size（内存边界）
  - 极小文件（单行或无数据）
  - 超大数值的解析
  - 特殊字符（换行、引号、分隔符）在CSV中

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - 本地文件系统访问
  - SQLite数据库文件
  - 压缩库（zlib, gzip）
  - TensorFlow运行时环境

- 需要mock/monkeypatch的部分：
  - tf.io.gfile.glob文件匹配
  - csv.reader解析逻辑
  - 数据库连接和查询执行
  - 文件I/O操作（open, read, close）
  - 压缩解压操作

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. CSV文件基本读取和类型推断
  2. 批处理功能验证（各种batch_size）
  3. 文件模式匹配和通配符支持
  4. SQL查询执行和结果解析
  5. 压缩文件读取和解压

- 可选路径（中/低优先级合并为一组列表）：
  - 并行读取性能测试
  - 内存使用监控
  - 错误忽略模式验证
  - 特殊字符和边界情况处理
  - 多文件合并读取
  - 类型转换边缘情况
  - 大文件分片读取
  - 缓存机制验证

- 已知风险/缺失信息（仅列条目，不展开）：
  - 类型推断逻辑测试覆盖不足
  - 并行读取线程安全性未明确
  - 内存使用特性需要基准
  - 大型数据集分片支持缺失
  - 压缩文件错误恢复机制不明确
  - V1/V2版本兼容性细节