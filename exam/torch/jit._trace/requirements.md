# torch.jit._trace 测试需求

## 1. 目标与范围
- 主要功能与期望行为：将Python函数或PyTorch模块转换为可执行的TorchScript ScriptFunction或ScriptModule，通过运行示例输入记录张量操作，生成优化后的JIT编译代码
- 不在范围内的内容：动态控制流（if语句、循环）、数据依赖的条件判断、未跟踪的外部依赖（I/O、全局变量访问）、复杂嵌套结构（自定义类）

## 2. 输入与约束
- 参数列表：
  - func (callable/torch.nn.Module)：必需，Python函数或PyTorch模块
  - example_inputs (tuple/torch.Tensor)：必需，追踪时使用的示例输入元组
  - check_trace (bool)：可选，默认True，是否验证追踪代码与原始函数输出一致
  - check_inputs (list of tuples)：可选，用于验证的额外输入参数列表
  - check_tolerance (float)：可选，默认1e-5，验证时的浮点数比较容差
  - strict (bool)：可选，默认True，是否在严格模式下运行追踪器
  - _force_outplace (bool)：内部参数，默认False
  - _module_class：内部参数，可选
  - _compilation_unit (CompilationUnit)：内部参数，默认<torch.jit.CompilationUnit object>
- 有效取值范围/维度/设备要求：参数和返回值必须是张量或包含张量的嵌套元组，支持CPU/GPU设备
- 必需与可选组合：func和example_inputs为必需参数，其余为可选
- 随机性/全局状态要求：无随机性要求，不修改全局状态

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 如果func是nn.Module或其forward方法：返回包含追踪代码的ScriptModule对象
  - 如果func是独立函数：返回ScriptFunction对象
  - 返回对象具有与原始模块相同的子模块和参数
- 容差/误差界：浮点数比较容差默认1e-5，可通过check_tolerance调整
- 状态变化或副作用检查点：可能产生TracerWarning，训练/评估模式行为固定为追踪时的模式

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：非张量输入、不支持的Python特性、动态控制流
- 边界值：空输入、None值、0长度张量、极端形状/数值、可变容器类型（list/dict）在严格模式下的处理

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：依赖PyTorch JIT编译机制，需要torch._C._jit_flatten和torch._C._jit_unflatten
- 需要mock/monkeypatch的部分：ONNXTracedModule包装机制、内部追踪函数、警告生成机制

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 基本函数追踪：简单张量操作函数
  2. 模块追踪：nn.Module及其forward方法
  3. 严格模式与非严格模式对比
  4. 验证机制测试：check_trace=True/False
  5. 多设备支持：CPU和GPU张量输入
- 可选路径（中/低优先级合并为一组列表）：
  - 复杂嵌套结构输入输出
  - 可变容器类型处理
  - 内部参数测试（_force_outplace, _module_class）
  - 训练/评估模式固定行为
  - 容差参数调整效果
  - 额外验证输入（check_inputs）使用
- 已知风险/缺失信息（仅列条目，不展开）：
  - 类型注解不完整
  - 支持的张量dtype和设备限制未明确
  - 复杂嵌套结构追踪支持说明缺失
  - 内部参数文档缺失
  - 非确定性操作处理未说明