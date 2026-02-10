# torch.jit._serialization 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试 TorchScript 模块的序列化（save）和反序列化（load）功能，确保模块能正确保存到文件/缓冲区并从文件/缓冲区正确加载
- 不在范围内的内容：模块训练过程、模型推理性能、非序列化相关的模块功能

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - save: m (ScriptModule), f (str/pathlib.Path/file-like), _extra_files (dict/None, 默认 None)
  - load: f (str/pathlib.Path/file-like), map_location (str/torch.device/None, 默认 None), _extra_files (dict/None, 默认 None)
- 有效取值范围/维度/设备要求：
  - ScriptModule 必须包含所有子模块为 ScriptModule 子类
  - 保存的模块不能调用原生 Python 函数
  - 文件对象需实现 write/flush（save）或 read/readline/tell/seek（load）
- 必需与可选组合：
  - save: m 和 f 必需，_extra_files 可选
  - load: f 必需，map_location 和 _extra_files 可选
- 随机性/全局状态要求：无随机性要求，加载时所有模块先到 CPU 再移动到原设备

## 3. 输出与判定
- 期望返回结构及关键字段：
  - save: 无返回值，成功执行 I/O 操作
  - load: 返回 ScriptModule 对象，保持原始模块结构和参数
- 容差/误差界（如浮点）：浮点参数在序列化前后应保持数值一致（允许极小浮点误差）
- 状态变化或副作用检查点：
  - 文件系统：save 创建文件，load 读取文件
  - 内存：缓冲区操作不影响其他内存区域
  - 设备：加载后模块应位于指定 map_location 设备

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非 ScriptModule 对象传递给 save
  - 文件不存在或权限不足
  - 无效的 map_location 参数
  - 不支持的文件对象接口
  - 包含原生 Python 函数的模块
- 边界值（空、None、0 长度、极端形状/数值）：
  - 空 ScriptModule
  - None 作为 f 参数
  - 空 _extra_files 字典
  - 超大模块序列化

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - 文件系统读写权限
  - GPU 设备（测试 GPU 相关功能）
  - torch._C_flatbuffer 模块（flatbuffer 功能）
- 需要 mock/monkeypatch 的部分：
  - 文件系统操作（权限错误、磁盘空间不足）
  - 设备可用性检查
  - flatbuffer 模块导入

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基本 save/load 循环：ScriptModule → 文件 → ScriptModule
  2. 缓冲区序列化：使用内存缓冲区而非文件
  3. 设备映射：CPU/GPU 设备间的正确映射
  4. 额外文件：_extra_files 参数的保存和加载
  5. 文件对象接口：支持文件对象而非路径
- 可选路径（中/低优先级合并为一组列表）：
  - flatbuffer 格式支持
  - 跨版本兼容性测试
  - 超大模块序列化性能
  - 并发访问文件
  - 异常恢复机制
- 已知风险/缺失信息（仅列条目，不展开）：
  - 文件对象接口的具体方法要求
  - flatbuffer 功能依赖 torch._C_flatbuffer 可用性
  - 跨版本操作符行为保持细节
  - 额外文件处理的具体限制
  - 内存缓冲区大小限制