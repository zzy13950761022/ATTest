# torch.nn.parallel.data_parallel 测试需求

## 1. 目标与范围
- 主要功能与期望行为：在多个GPU设备上并行评估神经网络模块，分散输入、并行执行、收集结果到指定设备
- 不在范围内的内容：DataParallel类的内部实现、分布式训练、梯度计算、模型训练过程

## 2. 输入与约束
- 参数列表：
  - module (torch.nn.Module): 神经网络模块，参数和缓冲区必须在device_ids[0]设备上
  - inputs (Tensor): 输入张量，支持任意形状，但需满足模块输入要求
  - device_ids (list[int|torch.device], 可选): GPU设备ID列表，默认使用所有可用GPU
  - output_device (int|torch.device, 可选): 输出设备，-1表示CPU，默认device_ids[0]
  - dim (int, 默认0): 分散/收集操作的维度
  - module_kwargs (dict, 可选): 传递给模块的关键字参数

- 有效取值范围/维度/设备要求：
  - device_ids必须为有效GPU设备ID列表
  - 模块参数和缓冲区必须在device_ids[0]设备上
  - dim必须在输入张量的有效维度范围内
  - 支持-1表示CPU作为输出设备

- 必需与可选组合：
  - module和inputs为必需参数
  - device_ids可选，默认使用所有可用GPU
  - output_device可选，默认device_ids[0]
  - module_kwargs可选，为空时使用空字典

- 随机性/全局状态要求：
  - 无随机性要求
  - 不修改模块参数，仅前向传播

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 返回Tensor，位于output_device指定设备
  - 形状与模块在单个设备上执行结果一致
  - 数值精度与单设备执行结果在容差范围内一致

- 容差/误差界（如浮点）：
  - 浮点误差容差：1e-6（相对误差）
  - 多设备并行结果应与单设备串行结果一致

- 状态变化或副作用检查点：
  - 模块参数和缓冲区保持不变
  - 不修改输入张量
  - 不改变CUDA设备状态

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - module非Module类型：TypeError
  - inputs非Tensor类型：TypeError
  - device_ids包含无效GPU ID：RuntimeError
  - 模块参数不在device_ids[0]设备上：RuntimeError
  - dim超出输入维度范围：IndexError
  - module_kwargs非字典类型：TypeError

- 边界值（空、None、0长度、极端形状/数值）：
  - device_ids为空列表：使用所有可用GPU
  - device_ids为None：使用所有可用GPU
  - inputs为空张量：模块处理空输入
  - 单GPU设备：直接调用模块
  - output_device=-1：输出到CPU
  - 极端大形状输入：内存限制测试
  - 极端小形状输入：维度有效性测试

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - CUDA设备可用性
  - 至少1个GPU设备
  - torch.cuda.is_available()返回True

- 需要mock/monkeypatch的部分：
  - `torch.nn.parallel.scatter_gather.scatter_kwargs`：输入分散
  - `torch.nn.parallel.scatter_gather.gather`：结果收集
  - `torch.cuda.device_count`：设备数量检测
  - `torch.cuda.current_device`：当前设备获取
  - `torch._C._get_available_device_types`：可用设备类型

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 单GPU设备正常执行
  2. 多GPU设备并行执行
  3. CPU作为输出设备（output_device=-1）
  4. 模块参数在不同设备上的验证
  5. 带module_kwargs的关键字参数传递

- 可选路径（中/低优先级合并为一组列表）：
  - 空device_ids列表处理
  - 不同dim值对分散/收集的影响
  - 极端形状输入处理
  - 混合精度输入测试
  - 嵌套模块结构测试
  - 梯度计算兼容性验证
  - 内存不足场景处理

- 已知风险/缺失信息（仅列条目，不展开）：
  - 文档缺少具体使用示例
  - 未明确输入张量形状要求
  - 未说明module_kwargs具体格式
  - 未说明dim参数具体影响
  - 未说明异常情况处理细节
  - 需要测试边界：空输入、单设备、多设备、CPU输出