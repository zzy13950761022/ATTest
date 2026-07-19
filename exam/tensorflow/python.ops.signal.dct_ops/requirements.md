# tensorflow.python.ops.signal.dct_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证 DCT 和 IDCT 函数正确实现离散余弦变换及其逆变换
  - 支持 DCT 类型 I、II、III、IV 的完整功能
  - 确保与 SciPy 的 `scipy.fftpack.dct` 数值一致性
  - 验证参数验证逻辑和错误处理
- 不在范围内的内容
  - 复数输入处理（仅支持 float32/float64）
  - axis 参数非 -1 的情况（当前仅支持 -1）
  - 非标准 DCT 类型（仅支持 1-4）

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - input: Tensor, shape [..., samples], float32/float64, 必需
  - type: int, 1/2/3/4, 默认 2
  - n: int/None, 正整数或 None, 默认 None
  - axis: int, 目前必须为 -1, 默认 -1
  - norm: str/None, None 或 'ortho', 默认 None
  - name: str/None, 操作名称, 默认 None
- 有效取值范围/维度/设备要求
  - input 维度必须 ≥ 1
  - Type-I DCT 要求 samples > 1
  - n 必须为正整数或 None
  - axis 目前仅支持 -1
  - Type-I DCT 不支持 'ortho' 归一化
- 必需与可选组合
  - input 为必需参数
  - type、n、axis、norm、name 为可选参数
  - idct 函数的 n 参数必须为 None
- 随机性/全局状态要求
  - 无随机性
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回 Tensor, shape [..., samples], 与输入相同数据类型
  - 输出维度与输入一致（n 参数影响除外）
- 容差/误差界（如浮点）
  - float32: 相对误差 ≤ 1e-5, 绝对误差 ≤ 1e-5
  - float64: 相对误差 ≤ 1e-10, 绝对误差 ≤ 1e-10
  - 与 SciPy 实现数值一致性验证
- 状态变化或副作用检查点
  - 无 I/O 操作
  - 无全局状态修改
  - 无随机数生成

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 非 float32/float64 输入类型 → ValueError
  - type 不在 {1,2,3,4} 范围内 → ValueError
  - Type-I DCT 使用 'ortho' 归一化 → ValueError
  - axis 不为 -1 → NotImplementedError
  - idct 函数 n 不为 None → ValueError
  - n 非正整数 → ValueError
- 边界值（空、None、0 长度、极端形状/数值）
  - 空张量输入 → 维度验证失败
  - Type-I DCT 输入 samples=1 → ValueError
  - n=0 或负整数 → ValueError
  - 极端大形状张量 → 内存/性能测试
  - 极端数值（inf, nan, 极大/极小值）→ 数值稳定性

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 运行时环境
  - SciPy（仅用于参考实现验证）
  - CPU/GPU 计算设备
- 需要 mock/monkeypatch 的部分
  - `tensorflow.python.ops.signal.fft_ops.rfft`（FFT 计算依赖）
  - `tensorflow.python.ops.signal.fft_ops.irfft`（逆 FFT 计算依赖）
  - `scipy.fftpack.dct`（参考实现验证）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 所有 DCT 类型（1-4）基本功能验证
  2. 浮点精度（float32/float64）兼容性测试
  3. 参数验证逻辑和异常处理
  4. 与 SciPy 实现的数值一致性
  5. n 参数截断/补零功能验证
- 可选路径（中/低优先级合并为一组列表）
  - 极端形状和大尺寸张量性能测试
  - 不同归一化选项组合测试
  - 批量处理多维度输入验证
  - 内存使用和泄漏检查
  - GPU 与 CPU 计算结果一致性
- 已知风险/缺失信息（仅列条目，不展开）
  - axis 参数仅支持 -1 的限制
  - idct 函数 n 参数必须为 None 的限制
  - Type-I DCT 归一化限制
  - 复数输入处理未明确
  - 极端数值稳定性未充分验证