# torch.ao.quantization.quantize 测试需求

## 1. 目标与范围
- 主要功能与期望行为：对浮点PyTorch模型执行后训练静态量化，包括校准准备、校准执行、量化转换三个步骤
- 不在范围内的内容：动态量化、量化感知训练、自定义量化配置、量化模型部署推理

## 2. 输入与约束
- 参数列表：
  - model：可评估的PyTorch浮点模型，无默认值
  - run_fn：校准函数，接受模型和run_args参数，无默认值
  - run_args：传递给run_fn的位置参数元组，无默认值
  - mapping：模块类型映射字典，默认使用get_default_static_quant_module_mappings()
  - inplace：布尔值，控制是否原地修改模型，默认False
- 有效取值范围/维度/设备要求：
  - model必须处于eval模式或可调用model.eval()
  - run_fn必须可调用且接受(model, *run_args)签名
  - mapping必须为字典或None
- 必需与可选组合：
  - model、run_fn、run_args为必需参数
  - mapping和inplace为可选参数
- 随机性/全局状态要求：
  - 函数内部调用torch._C._log_api_usage_once记录API使用
  - 非原地操作时使用copy.deepcopy复制模型

## 3. 输出与判定
- 期望返回结构及关键字段：量化后的PyTorch模型，结构与输入模型相同但包含量化层
- 容差/误差界（如浮点）：量化模型与原始浮点模型在相同输入下输出差异应在可接受范围内
- 状态变化或副作用检查点：
  - 当inplace=True时，原始模型被修改
  - 当inplace=False时，原始模型保持不变
  - 模型始终被设置为eval模式

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非PyTorch模型作为model参数
  - 不可调用的run_fn函数
  - 无效的mapping字典结构
  - 训练模式的模型（未调用model.eval()）
- 边界值（空、None、0长度、极端形状/数值）：
  - model为None或空模型
  - run_args为空元组
  - mapping为None（使用默认映射）
  - 极端大模型的内存限制

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - PyTorch量化模块的可用性
  - 默认静态量化模块映射函数
  - prepare和convert辅助函数
- 需要mock/monkeypatch的部分：
  - torch._C._log_api_usage_once调用
  - copy.deepcopy操作
  - get_default_static_quant_module_mappings函数
  - prepare和convert函数调用

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 基本浮点模型量化流程验证
  2. inplace=True与False的行为差异
  3. 自定义mapping参数的正确应用
  4. 校准函数run_fn的正确调用和参数传递
  5. 量化后模型结构与功能完整性
- 可选路径（中/低优先级合并为一组列表）：
  - 不同模型架构的兼容性测试
  - 复杂run_args参数组合
  - 边界情况：空模型、最小模型
  - 性能基准测试（可选）
  - 量化精度验证（可选）
- 已知风险/缺失信息（仅列条目，不展开）：
  - 参数类型注解缺失
  - run_fn函数签名要求不明确
  - 支持的模型层类型未说明
  - 量化配置选项未文档化
  - 错误处理机制未说明