# torch.serialization 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试PyTorch序列化模块的保存/加载功能，验证张量、模型、任意对象的正确序列化与反序列化，保持存储共享关系，支持设备映射
- 不在范围内的内容：自定义pickle协议实现细节、第三方pickle模块的内部逻辑、文件系统权限错误处理

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - save: obj(object), f(FILE_LIKE), pickle_module(Any,默认pickle), pickle_protocol(int,默认2), _use_new_zipfile_serialization(bool,默认True)
  - load: f(FILE_LIKE), map_location(MAP_LOCATION,默认None), pickle_module(Any,默认None), weights_only(bool,默认False), **pickle_load_args
- 有效取值范围/维度/设备要求：
  - pickle_protocol: 0-4（兼容Python版本）
  - map_location: 'cpu'、'cuda:device_id'、torch.device对象、字典、可调用函数
  - FILE_LIKE: 文件路径字符串、BytesIO、文件对象（需实现相应方法）
- 必需与可选组合：
  - save: obj和f必需，其他可选
  - load: f必需，其他可选
- 随机性/全局状态要求：无随机性要求，register_package可能修改全局包注册表

## 3. 输出与判定
- 期望返回结构及关键字段：
  - save: 无返回值（None），文件被创建/覆盖
  - load: 返回保存的原始对象，类型与保存时一致
- 容差/误差界（如浮点）：浮点张量反序列化后数值误差<1e-7
- 状态变化或副作用检查点：
  - 文件系统：文件创建、修改时间、大小变化
  - 内存：存储共享关系保持
  - 设备：张量设备位置正确映射

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 无效文件路径：FileNotFoundError/OSError
  - 不支持的文件对象：AttributeError（缺少read/write方法）
  - 不兼容的pickle_module：RuntimeError
  - weights_only模式下加载不安全对象：RuntimeError
  - 损坏的序列化文件：EOFError/UnpicklingError
- 边界值（空、None、0长度、极端形状/数值）：
  - 空张量（shape包含0）
  - None对象保存/加载
  - 极大张量（内存边界）
  - 特殊数值（inf、nan、-0.0）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - 文件系统读写权限
  - CUDA设备（测试GPU映射时）
  - 临时文件存储空间
- 需要mock/monkeypatch的部分：
  - `torch.serialization._is_zipfile`（测试文件格式检测）
  - `pickle.dump`/`pickle.load`（测试pickle模块替换）
  - `io.BytesIO`/`io.open`（测试文件类对象）
  - `torch.cuda.is_available`（测试设备可用性）
  - `os.path.exists`/`os.remove`（测试文件操作）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 基本张量保存加载（CPU/GPU）
  2. weights_only安全模式验证
  3. map_location设备映射功能
  4. 新旧zip文件格式兼容性
  5. 存储共享关系保持
- 可选路径（中/低优先级合并为一组列表）：
  - 自定义pickle模块替换
  - 复杂嵌套对象序列化
  - 大文件/内存压力测试
  - 并发访问文件锁
  - 网络文件系统路径
  - 特殊字符文件名处理
- 已知风险/缺失信息（仅列条目，不展开）：
  - register_package函数文档不完整
  - pickle_module参数类型信息不足
  - weights_only模式具体限制未详细说明
  - map_location复杂类型变体覆盖不全
  - 错误处理边界案例文档缺失