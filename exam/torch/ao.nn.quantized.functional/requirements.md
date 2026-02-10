# torch.ao.nn.quantized.functional 测试需求

## 1. 目标与范围
- 验证量化神经网络函数正确执行量化张量操作
- 确保输入量化参数正确传播到输出
- 验证函数行为与 torch.nn.functional 对应函数一致（量化版本）
- 不在范围内的内容：非量化张量操作、自定义量化方案、训练过程

## 2. 输入与约束
- 参数列表（以 conv2d 为例）：
  - input: torch.Tensor, 形状 (N, C, H, W), 必须是 torch.quint8 类型
  - weight: torch.Tensor, 形状 (out_channels, in_channels/groups, kH, kW), 必须是 torch.qint8 类型
  - bias: torch.Tensor, 形状 (out_channels), 必须是 torch.float 类型
  - stride: int/tuple, 默认 1
  - padding: int/tuple, 默认 0
  - dilation: int/tuple, 默认 1
  - groups: int, 默认 1
  - padding_mode: str, 仅支持 'zeros'
  - scale: float, 默认 1.0
  - zero_point: int, 默认 0
  - dtype: torch.dtype, 默认 torch.quint8

- 有效取值范围/维度/设备要求：
  - 输入张量必须满足 input.is_quantized == True
  - conv2d 需要 4D 输入张量
  - groups 必须能整除 in_channels
  - 仅支持零填充模式
  - 量化参数 scale > 0

- 必需与可选组合：
  - input, weight 为必需参数
  - bias 为可选参数（可传 None）
  - stride, padding, dilation, groups 有默认值
  - scale, zero_point 用于指定输出量化参数

- 随机性/全局状态要求：
  - 无随机性操作
  - 无全局状态修改
  - 确定性操作

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 返回量化张量 torch.Tensor
  - 输出形状遵循标准卷积公式
  - 输出具有量化属性（is_quantized == True）
  - 输出使用指定或继承的 scale/zero_point

- 容差/误差界（如浮点）：
  - 量化误差在可接受范围内
  - 与浮点版本结果对比误差 < 1e-3
  - 量化参数传播误差 < 1e-6

- 状态变化或副作用检查点：
  - 无 I/O 操作
  - 无全局状态修改
  - 输入张量不被修改

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非量化张量输入触发 ValueError
  - 错误的 dtype 组合触发 ValueError
  - 无效的 padding_mode 触发 NotImplementedError
  - 维度不匹配触发 RuntimeError
  - groups 不能整除 in_channels 触发 ValueError

- 边界值（空、None、0 长度、极端形状/数值）：
  - 空张量或零维度输入
  - scale <= 0 的量化参数
  - 极端大的 zero_point 值
  - 最小/最大量化值边界
  - 单元素张量输入

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - 依赖 torch.ops.quantized.* 底层操作
  - 需要量化操作支持
  - 无网络/文件依赖

- 需要 mock/monkeypatch 的部分：
  - torch.ops.quantized.* 调用可 mock 验证参数传递
  - 量化辅助函数可 mock 验证正确性
  - 异常路径测试需要 mock 底层错误

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基本量化卷积操作正确性验证
  2. 量化参数正确传播到输出
  3. 不同类型量化张量组合验证
  4. 边界形状和极端值处理
  5. 错误输入触发正确异常

- 可选路径（中/低优先级合并为一组列表）：
  - 不同 stride/padding/dilation 组合
  - groups 参数的各种有效值
  - 多种量化 scale/zero_point 组合
  - 与浮点版本结果对比
  - 性能基准测试（linear 函数权重打包开销）
  - 已弃用函数（upsample 系列）的兼容性

- 已知风险/缺失信息（仅列条目，不展开）：
  - 模块包含 25+ 函数需要分别测试
  - 部分函数已弃用但仍在模块中
  - 缺少完整类型注解
  - 设备兼容性（CPU/GPU）未明确说明
  - 某些量化数据类型组合可能未覆盖
  - 错误处理路径可能不完整