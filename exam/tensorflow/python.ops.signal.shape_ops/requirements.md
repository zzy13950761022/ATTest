# tensorflow.python.ops.signal.shape_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证 `frame` 函数正确将信号张量沿指定轴展开为帧
  - 测试滑动窗口（大小 `frame_length`，步长 `frame_step`）的分帧逻辑
  - 验证 `pad_end` 和 `pad_value` 参数对末尾填充的影响
  - 确保负轴索引的正确处理
- 不在范围内的内容
  - 不测试模块内部辅助函数 `_infer_frame_shape` 的独立行为
  - 不验证 TensorFlow 基础张量操作（如 `strided_slice`, `gather`）的正确性
  - 不涉及信号处理算法（如 FFT、滤波）的数学正确性

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `signal`: Tensor，秩≥1，形状 `[..., samples, ...]`
  - `frame_length`: int/Tensor 标量，无默认值
  - `frame_step`: int/Tensor 标量，无默认值
  - `pad_end`: bool，默认 `False`
  - `pad_value`: scalar Tensor，默认 `0`
  - `axis`: int Tensor，默认 `-1`
  - `name`: str，默认 `None`
- 有效取值范围/维度/设备要求
  - `signal` 秩必须至少为1
  - `frame_length`, `frame_step`, `pad_value`, `axis` 必须是标量
  - `axis` 支持负索引（-1表示最后一个轴）
  - 所有参数支持 CPU/GPU 设备
- 必需与可选组合
  - `signal`, `frame_length`, `frame_step` 为必需参数
  - `pad_end`, `pad_value`, `axis`, `name` 为可选参数
  - `pad_value` 仅在 `pad_end=True` 时生效
- 随机性/全局状态要求
  - 无随机性操作
  - 无全局状态修改

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回 Tensor，形状 `[..., num_frames, frame_length, ...]`
  - 帧数计算：
    - `pad_end=False`: `num_frames = 1 + (N - frame_length) // frame_step`
    - `pad_end=True`: `num_frames = -(-N // frame_step)`（向上取整除法）
- 容差/误差界（如浮点）
  - 浮点类型（float16/32/64）允许机器精度误差
  - 整数类型要求精确匹配
  - 填充值 `pad_value` 必须精确匹配
- 状态变化或副作用检查点
  - 无 I/O 操作
  - 无全局变量修改
  - 输入张量保持不变

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - `signal` 秩为0（标量）应触发异常
  - `frame_length` 或 `frame_step` 为非标量应触发异常
  - `axis` 超出有效范围（`-rank <= axis < rank`）应触发异常
  - `frame_length` 或 `frame_step` 非正整数应触发异常
  - `pad_value` 与 `signal` dtype 不兼容应触发异常
- 边界值（空、None、0 长度、极端形状/数值）
  - `signal` 沿 `axis` 维度长度为0
  - `frame_length` > 信号沿 `axis` 维度长度
  - `frame_step` = 0
  - `frame_length` = 0
  - 极大 `frame_length` 或 `frame_step` 值
  - 负 `frame_length` 或 `frame_step`
  - `axis` = 0（第一个轴）和 `axis` = -1（最后一个轴）边界

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 运行时环境
  - 无网络或文件系统依赖
- 需要 mock/monkeypatch 的部分
  - `tensorflow.python.ops.signal.shape_ops.util_ops.gcd`（最大公约数计算）
  - `tensorflow.python.ops.array_ops.strided_slice`
  - `tensorflow.python.ops.array_ops.gather`
  - `tensorflow.python.ops.math_ops.floordiv`
  - `tensorflow.python.framework.tensor_util.constant_value`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 基本分帧功能：正常信号，`pad_end=False`，默认 `axis=-1`
  2. 末尾填充：`pad_end=True`，验证填充值和帧数计算
  3. 负轴索引：`axis=-1` 和 `axis=-2` 的正确处理
  4. 边界条件：`frame_length` > 信号长度，`pad_end=False/True`
  5. 数据类型：float16/32/64，int8/16/32/64 的兼容性
- 可选路径（中/低优先级合并为一组列表）
  - 多维信号（秩≥3）的分帧
  - 不同设备（CPU/GPU）的一致性
  - 大张量性能（内存使用验证）
  - `frame_length` 和 `frame_step` 为 Tensor 对象
  - `pad_value` 为不同标量类型
  - `name` 参数的正确传递
- 已知风险/缺失信息（仅列条目，不展开）
  - 未明确支持的 dtype 完整列表
  - `pad_value` 类型转换规则
  - 极端大值（接近 int64 上限）的处理
  - 稀疏张量的支持情况
  - 梯度计算正确性