# torch.onnx.utils 测试需求

## 1. 目标与范围
- 主要功能：将 PyTorch 模型转换为 ONNX 格式，支持多种导出配置和模型类型
- 期望行为：正确序列化模型为 ONNX 协议缓冲区，处理各种输入格式和导出选项
- 不在范围内：ONNX 模型推理验证、第三方 ONNX 运行时集成、模型性能基准测试

## 2. 输入与约束
- 参数列表：
  - model: torch.nn.Module/torch.jit.ScriptModule/torch.jit.ScriptFunction
  - args: Tuple[Any]/torch.Tensor/带命名参数的元组
  - f: str/io.BytesIO，输出目标
  - export_params: bool, default=True
  - verbose: bool, default=False
  - training: _C_onnx.TrainingMode, default=EVAL
  - input_names: Optional[Sequence[str]], default=None
  - output_names: Optional[Sequence[str]], default=None
  - operator_export_type: _C_onnx.OperatorExportTypes, default=ONNX
  - opset_version: Optional[int], default=None (7-16范围)
  - do_constant_folding: bool, default=True
  - dynamic_axes: Optional[Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]], default=None
  - keep_initializers_as_inputs: Optional[bool], default=None
  - custom_opsets: Optional[Mapping[str, int]], default=None
  - export_modules_as_functions: Union[bool, Collection[Type[torch.nn.Module]]], default=False

- 有效取值范围：
  - opset_version: 7-16整数
  - 动态轴定义：字典或列表格式
  - 训练模式：EVAL/TRAINING/Preserve

- 必需组合：
  - 非 ScriptModule/Function 模型自动通过 torch.jit.trace 转换
  - 导出训练模式需要 opset_version >= 12

- 随机性/全局状态：
  - 依赖 GLOBALS 全局状态管理导出配置
  - torch.jit.trace 引入随机性

## 3. 输出与判定
- 期望返回：export()返回None，成功写入文件或缓冲区
- export_to_pretty_string()返回UTF-8字符串格式的ONNX模型
- is_in_onnx_export()返回布尔值指示导出状态
- 容差：浮点数值精度误差在1e-6范围内
- 状态变化：文件系统写入验证，缓冲区内容完整性检查
- 副作用：全局状态GLOBALS正确更新和恢复

## 4. 错误与异常场景
- 非法输入：非模型对象、无效文件路径、不支持的数据类型
- 维度错误：输入输出形状不匹配、动态轴定义冲突
- 类型错误：参数类型不符合声明、Python对象无法序列化
- 边界值：空模型、None输入、0长度张量、极端形状(如1x1x1)
- 数值边界：NaN、Inf、极大/极小浮点值
- 版本冲突：opset_version超出7-16范围、不支持的算子版本
- 资源限制：内存不足、磁盘空间不足、文件权限错误

## 5. 依赖与环境
- 外部依赖：ONNX库、protobuf、文件系统访问
- 设备要求：CPU/GPU张量支持，设备间数据传输
- 需要mock部分：
  - `torch.jit.trace`：控制跟踪行为
  - `torch.onnx._internals._model_to_graph`：图转换过程
  - `io.open`：文件I/O操作
  - `GLOBALS`：全局状态管理
  - `_beartype.beartype`：运行时类型检查
  - `torch.nn.Module`的forward方法调用

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. 三种模型类型(nn.Module/ScriptModule/ScriptFunction)的基本导出
  2. args三种格式(元组/张量/命名参数)的正确处理
  3. opset_version边界值(7,16)和默认值的兼容性
  4. 动态轴配置的字典和列表格式支持
  5. 文件路径和BytesIO两种输出目标的正确写入

- 可选路径（中/低优先级）：
  - 不同operator_export_type(ONNX/ONNX_ATEN/ONNX_ATEN_FALLBACK)的行为差异
  - export_params=False时的参数分离导出
  - training模式切换与opset_version>=12的关联
  - do_constant_folding优化效果验证
  - custom_opsets自定义算子集配置
  - export_modules_as_functions模块函数化导出
  - keep_initializers_as_inputs不同设置的影响
  - verbose详细输出模式的内容验证
  - 复杂模型结构(嵌套、循环、条件)的导出能力

- 已知风险/缺失信息：
  - 动态控制流支持有限（与torch.jit.trace相同限制）
  - 某些导出选项依赖特定构建配置（如Caffe2支持）
  - 缺少具体张量形状和dtype的详细约束说明
  - 模型参数序列化的内存使用边界
  - 大模型导出的性能和时间消耗
  - 多线程/多进程环境下的并发安全性