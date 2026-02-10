# torch.nn.modules.pooling 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证池化层（MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveMaxPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d, FractionalMaxPool2d/3d, LPPool1d/2d, MaxUnpool1d/2d/3d）的正确实例化、前向传播、输出形状计算
- 不在范围内的内容：池化层的反向传播梯度计算、训练过程中的参数更新、与其他模块的组合集成

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - kernel_size (_size_1_t/_size_2_t/_size_3_t)：滑动窗口大小，必须 > 0
  - stride (Optional[_size_1_t]): 滑动步长，默认等于 kernel_size，必须 > 0
  - padding (_size_1_t): 隐式填充点数，默认 0，必须 >= 0 且 <= kernel_size/2
  - dilation (_size_1_t): 窗口内元素间距，默认 1，必须 > 0
  - return_indices (bool): 是否返回最大值索引，默认 False
  - ceil_mode (bool): 使用 ceil 而非 floor 计算输出形状，默认 False
  - output_size (int/tuple): 自适应池化的目标输出尺寸
  - output_ratio (float/tuple): 分数最大池化的输出比例，0-1之间

- 有效取值范围/维度/设备要求：
  - 输入形状：(N, C, L_in) 或 (C, L_in) 对于1D池化，对应维度类推
  - kernel_size, stride, padding, dilation 必须为正整数
  - padding <= kernel_size/2
  - 支持 CPU 和 CUDA 设备

- 必需与可选组合：
  - kernel_size 必需，其他参数可选
  - return_indices 仅适用于最大池化类
  - ceil_mode 影响输出形状计算

- 随机性/全局状态要求：
  - FractionalMaxPool2d/3d 具有随机池化区域选择
  - 无全局状态副作用

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 池化层实例，forward 返回 Tensor
  - return_indices=True 时返回 (output, indices) 元组
  - indices 为最大值在原输入中的位置索引

- 容差/误差界（如浮点）：
  - 浮点计算误差在 1e-6 范围内可接受
  - 平均池化结果应与手动计算一致
  - 最大池化应正确识别最大值位置

- 状态变化或副作用检查点：
  - 无持久状态变化
  - 无文件/网络 I/O
  - 无全局变量修改

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - kernel_size <= 0 触发 ValueError
  - stride <= 0 触发 ValueError
  - padding < 0 或 padding > kernel_size/2 触发 ValueError
  - dilation <= 0 触发 ValueError
  - 输入维度不匹配（如2D输入给1D池化）触发 RuntimeError
  - output_ratio 超出 [0,1] 范围触发 ValueError

- 边界值（空、None、0 长度、极端形状/数值）：
  - 输入 Tensor 为空或零长度维度
  - kernel_size 大于输入尺寸
  - 极端大尺寸输入（内存边界）
  - 极端小尺寸输入（1x1）
  - 负无穷填充值的处理
  - ceil_mode=True 时滑动窗口超出边界的情况

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - torch 库及其 functional 模块
  - CUDA 设备（可选测试）
  - 无网络/文件系统依赖

- 需要 mock/monkeypatch 的部分：
  - FractionalMaxPool 的随机数生成器
  - torch.nn.functional 中的池化函数调用
  - 设备检测和内存分配

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基本池化类正确实例化和前向传播
  2. 输出形状计算公式验证（floor/ceil模式）
  3. 最大池化的 return_indices 功能
  4. 自适应池化输出尺寸控制
  5. 不同维度（1D/2D/3D）池化一致性

- 可选路径（中/低优先级合并为一组列表）：
  - 分数最大池化的随机性行为
  - LPPool 的 p-norm 计算
  - MaxUnpool 的反池化操作
  - 极端形状和边界条件
  - 不同设备（CPU/CUDA）一致性
  - 批量处理和多通道输入
  - 填充和膨胀参数组合

- 已知风险/缺失信息（仅列条目，不展开）：
  - dilation 参数的最大值约束未明确
  - 类型别名 _size_1_t 等的具体定义
  - 分数最大池化的随机性控制机制
  - 自适应池化中 output_size 的边界处理
  - 不同池化类的参数默认值差异