# tensorflow.python.ops.signal.fft_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证复数FFT（fft/ifft）1D/2D/3D变换的正确性
  - 验证实数FFT（rfft/irfft）1D/2D/3D变换的正确性
  - 验证频谱移位（fftshift/ifftshift）与NumPy等效性
  - 确保梯度计算正确性，特别是Hermitian对称性处理
- 不在范围内的内容
  - 性能基准测试和内存使用分析
  - 不同硬件设备（CPU/GPU/TPU）的性能差异
  - 与其他FFT库（如FFTW）的兼容性比较

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - x: Tensor，任意数据类型，无默认值
  - axes: int/tuple/None，默认None表示所有轴
  - name: str/None，默认None
  - fft_length: int，仅rfft/irfft使用
- 有效取值范围/维度/设备要求
  - 支持1D、2D、3D张量输入
  - 支持负轴索引（自动转换为正索引）
  - 实数FFT（rfft）仅返回fft_length/2+1个唯一分量
  - 无特定设备限制，但需测试CPU环境
- 必需与可选组合
  - x为必需参数
  - axes和name为可选参数
  - fft_length为rfft/irfft必需参数
- 随机性/全局状态要求
  - 无随机性操作
  - 无全局状态修改
  - 无I/O操作

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回Tensor，保持输入数据类型
  - 复数FFT返回复数类型张量
  - 实数FFT返回实数类型张量
- 容差/误差界（如浮点）
  - 与NumPy结果比较，相对误差<1e-6
  - 梯度计算误差<1e-4
  - 循环移位精度误差<1e-12
- 状态变化或副作用检查点
  - 无副作用
  - 输入张量不被修改
  - 无文件系统或网络操作

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 非Tensor输入触发TypeError
  - 无效轴索引触发ValueError
  - 不支持的数据类型触发TypeError
  - fft_length小于输入长度触发ValueError
- 边界值（空、None、0长度、极端形状/数值）
  - 空张量输入
  - 零长度张量
  - 单元素张量
  - 极大形状张量（内存边界）
  - 极端数值（inf, nan, 极大/极小值）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - 无外部资源依赖
  - 无网络连接需求
  - 无文件系统访问
- 需要mock/monkeypatch的部分
  - `tensorflow.python.ops.manip_ops.roll`（fftshift依赖）
  - `tensorflow.python.ops.gen_spectral_ops`（FFT底层实现）
  - `tensorflow.python.ops.math_ops`（数学运算）
  - `tensorflow.python.framework.ops`（张量操作）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. fft/ifft正向和逆向变换的互逆性验证
  2. rfft/irfft实数变换的正确性和梯度计算
  3. fftshift/ifftshift与NumPy的等效性验证
  4. 不同轴参数（None/int/tuple）的处理正确性
  5. 边界情况：空张量、单元素、极端形状

- 可选路径（中/低优先级合并为一组列表）
  - 不同数据类型（float32/float64/complex64/complex128）的兼容性
  - 负轴索引的自动转换验证
  - 大尺寸张量的内存使用和性能
  - 与其他TensorFlow信号处理模块的集成
  - 批处理模式下的正确性

- 已知风险/缺失信息（仅列条目，不展开）
  - 部分函数通过gen_spectral_ops实现，源码可见性有限
  - 梯度函数实现复杂，Hermitian对称性处理细节不明确
  - 未明确支持的设备类型限制
  - 缺少详细的性能约束说明
  - 实数FFT返回分量数目的数学依据未详细说明