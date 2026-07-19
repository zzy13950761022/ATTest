# torch.nn.modules.activation 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证28个激活函数类（ReLU、Sigmoid、Tanh、Softmax等）的正确实现
  - 确保所有激活函数继承自torch.nn.Module，支持标准神经网络层接口
  - 验证输入张量形状保持不变，仅进行逐元素变换
  - 测试inplace操作的内存优化行为（如ReLU、LeakyReLU）
  - 验证MultiheadAttention的复杂注意力机制

- 不在范围内的内容
  - torch.nn.functional模块的底层实现细节
  - 激活函数的数学理论证明
  - 性能基准测试和优化
  - 不同PyTorch版本的兼容性差异

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - 通用输入：任意维度张量（*表示任意维度）
  - ReLU：inplace (bool, 默认False)
  - Softmax：dim (int, 可选，默认None)
  - Hardtanh：min_val (float, 默认-1.0), max_val (float, 默认1.0), inplace (bool, 默认False)
  - MultiheadAttention：embed_dim (int), num_heads (int), dropout (float, 默认0.0), bias (bool, 默认True), add_bias_kv (bool, 默认False), add_zero_attn (bool, 默认False), kdim (int, 可选), vdim (int, 可选)

- 有效取值范围/维度/设备要求
  - 所有函数支持CPU和GPU设备
  - Hardtanh要求max_val > min_val
  - MultiheadAttention要求embed_dim能被num_heads整除
  - Softmax的dim参数必须在输入张量维度范围内

- 必需与可选组合
  - MultiheadAttention：embed_dim和num_heads为必需参数
  - 大多数激活函数无必需参数（使用默认构造函数）
  - inplace参数为可选，默认False

- 随机性/全局状态要求
  - RReLU在训练模式（model.train()）下有随机行为
  - 测试时需要区分训练和评估模式
  - 无全局状态修改，无I/O操作

## 3. 输出与判定
- 期望返回结构及关键字段
  - 所有激活函数返回与输入形状相同的张量
  - MultiheadAttention返回元组：(output, attention_weights)
  - 输出值范围符合数学定义（如Sigmoid: (0, 1), Tanh: (-1, 1)）

- 容差/误差界（如浮点）
  - 浮点比较使用torch.allclose，rtol=1e-5, atol=1e-8
  - 梯度计算误差容忍度适当放宽
  - 边缘情况（如接近0的输入）需要特殊容差处理

- 状态变化或副作用检查点
  - inplace=True时，输入张量被修改
  - inplace=False时，输入张量保持不变
  - 训练模式切换影响RReLU的随机行为
  - 无持久化状态变化

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 非张量输入触发TypeError
  - 无效dim参数触发IndexError
  - Hardtanh的max_val <= min_val触发ValueError
  - MultiheadAttention的embed_dim不能被num_heads整除触发ValueError
  - 不支持的数据类型触发RuntimeError

- 边界值（空、None、0长度、极端形状/数值）
  - 空张量输入（torch.tensor([])）
  - 标量输入（0维张量）
  - 极端大/小数值（inf, -inf, NaN）
  - 零值输入（测试ReLU(0)等边界）
  - 极大形状张量（内存边界测试）
  - 负维度参数

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - PyTorch库依赖
  - CUDA设备（可选，用于GPU测试）
  - 无网络/文件系统依赖

- 需要mock/monkeypatch的部分
  - torch.nn.functional函数调用（验证正确转发）
  - 随机数生成器（控制RReLU的随机性）
  - CUDA可用性检查（设备兼容性测试）
  - 内存分配（inplace操作验证）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. 所有激活函数的基础正向传播正确性
  2. ReLU/Sigmoid/Tanh/Softmax的梯度计算验证
  3. inplace参数的内存行为测试
  4. MultiheadAttention的标准注意力计算
  5. 极端输入值（inf, NaN, 大数值）处理

- 可选路径（中/低优先级合并为一组列表）
  - RReLU训练/评估模式切换
  - Hardtanh参数边界验证
  - 不同数据类型的支持（float16, float32, float64）
  - 批量处理和大形状张量
  - 序列化/反序列化（torch.save/load）
  - 设备间转移（CPU↔GPU）
  - 与其他nn.Module的组合使用

- 已知风险/缺失信息（仅列条目，不展开）
  - MultiheadAttention优化路径条件复杂
  - Softmax数值稳定性（大数值溢出）
  - 部分参数类型注解缺失
  - RReLU随机性难以完全控制
  - 缺少性能退化检测机制