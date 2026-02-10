# torch.nn.modules.conv 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试 Conv1d、Conv2d、Conv3d 及其转置版本的正确性，包括参数验证、前向传播、输出形状计算、权重初始化
- 不在范围内的内容：底层卷积算法实现细节、GPU 特定优化、性能基准测试、与其他框架的兼容性

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - in_channels (int, 必需)：正整数
  - out_channels (int, 必需)：正整数
  - kernel_size (int/tuple, 必需)：正整数或元组
  - stride (int/tuple, 默认 1)：正整数或元组
  - padding (int/tuple/str, 默认 0)：整数/元组或 'valid'/'same'
  - dilation (int/tuple, 默认 1)：正整数或元组
  - groups (int, 默认 1)：正整数
  - bias (bool, 默认 True)：布尔值
  - padding_mode (str, 默认 'zeros')：'zeros'/'reflect'/'replicate'/'circular'
- 有效取值范围/维度/设备要求：
  - groups > 0 且必须整除 in_channels 和 out_channels
  - padding='same' 时 stride 必须为 1
  - 支持 CPU 和 CUDA 设备
  - 支持 float32、float64、complex32、complex64、complex128 数据类型
- 必需与可选组合：in_channels、out_channels、kernel_size 为必需参数
- 随机性/全局状态要求：权重初始化使用均匀分布 U(-√k, √k)，其中 k = groups/(C_in * ∏kernel_size)

## 3. 输出与判定
- 期望返回结构及关键字段：返回可调用模块实例，forward 方法返回 Tensor
- 容差/误差界（如浮点）：浮点计算容差 1e-5，复数计算容差 1e-7
- 状态变化或副作用检查点：权重和偏置正确初始化，模块状态可保存/加载

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - groups 不整除 in_channels/out_channels 时抛出 ValueError
  - padding_mode 不在允许集合中时抛出 ValueError
  - padding='same' 且 stride != 1 时抛出 ValueError
  - 输入 Tensor 维度不匹配时抛出 RuntimeError
- 边界值（空、None、0 长度、极端形状/数值）：
  - in_channels=0 或 out_channels=0 时异常
  - kernel_size=0 时异常
  - 极大形状（接近内存限制）测试
  - 极小形状（1x1）测试

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：PyTorch 库、CUDA 运行时（可选）
- 需要 mock/monkeypatch 的部分：
  - torch.nn.functional.conv* 函数调用
  - 随机数生成器用于确定性测试
  - CUDA 可用性检测

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基本 Conv2d 实例化与前向传播
  2. groups 参数验证与整除性检查
  3. padding='same' 与 stride=1 的组合
  4. 不同 padding_mode 的正确处理
  5. 权重初始化分布验证
- 可选路径（中/低优先级合并为一组列表）：
  - 转置卷积类测试
  - 1D 和 3D 卷积测试
  - 复数数据类型支持
  - 极端形状和大尺寸输入
  - 设备间迁移（CPU↔CUDA）
  - 序列化/反序列化测试
  - 梯度计算正确性
- 已知风险/缺失信息（仅列条目，不展开）：
  - padding='same' 输出形状的精确描述缺失
  - 复数数据类型支持的具体限制未详细说明
  - CUDA 非确定性算法行为
  - 权重初始化公式中的浮点精度问题