# tensorflow.python.ops.gen_spectral_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证频谱操作模块（FFT/IFFT/RFFT/IRFFT及其2D/3D/批处理版本）的正确性、边界处理和异常响应
- 不在范围内的内容：底层C++实现细节、梯度计算验证、性能基准测试、非标准复数类型支持

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - input: Tensor[float32/float64/complex64/complex128]，形状因函数而异
  - fft_length: Tensor[int32]，部分函数必需，控制变换长度
  - name: str/None，操作名称，可选
  - Tcomplex/Treal: dtype参数，有默认值（complex64/float32）

- 有效取值范围/维度/设备要求：
  - 输入张量维度：1D变换至少1维，2D变换至少2维，3D变换至少3维
  - 批处理函数支持批量维度
  - 仅支持CPU/GPU设备，不支持TPU特殊处理
  - fft_length必须为正整数

- 必需与可选组合：
  - FFT/IFFT系列：仅需input参数
  - RFFT/IRFFT系列：必需input和fft_length参数
  - 所有函数：name参数可选

- 随机性/全局状态要求：
  - 无随机性要求
  - 无全局状态依赖
  - 操作应为纯函数

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 复数变换（FFT/IFFT）：返回复数张量，保持输入形状（最后一维可能变化）
  - 实数变换（RFFT）：返回复数张量，形状为[..., fft_length//2+1]
  - 逆实数变换（IRFFT）：返回实数张量，形状由fft_length决定
  - 批处理版本：保持批量维度不变

- 容差/误差界（如浮点）：
  - 浮点误差：相对误差<1e-6，绝对误差<1e-8
  - 复数相位误差：角度误差<1e-6弧度
  - 逆变换还原误差：FFT(IFFT(x)) ≈ x，误差<1e-6

- 状态变化或副作用检查点：
  - 无文件系统操作
  - 无网络访问
  - 无全局变量修改
  - 输入张量应保持不变

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非数值类型输入：触发TypeError
  - 不支持的数据类型：触发ValueError
  - 维度不足：触发ValueError（如1D输入给fft2d）
  - 无效fft_length：非正整数触发ValueError
  - 复数输入给RFFT：触发ValueError

- 边界值（空、None、0长度、极端形状/数值）：
  - 空张量：触发ValueError
  - None输入：触发TypeError
  - 零长度维度：触发ValueError
  - 极大fft_length：内存不足时触发ResourceExhaustedError
  - 极小fft_length（1）：验证最小有效值
  - 奇数fft_length：验证IRFFT特殊处理
  - 极端数值（inf/nan）：验证传播行为

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow库依赖
  - CUDA/cuDNN（GPU测试时）
  - 无网络/文件系统依赖

- 需要mock/monkeypatch的部分：
  - `tensorflow.python.framework.ops.device`：设备分配
  - `tensorflow.python.framework.dtypes`：数据类型验证
  - `tensorflow.python.ops.gen_spectral_ops._execute`：底层执行
  - `tensorflow.python.eager.context`：执行模式切换
  - `tensorflow.python.ops.gen_spectral_ops._op_def_library`：操作定义

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 基本FFT/IFFT变换的正确性和可逆性
  2. RFFT/IRFFT的实数-复数转换和长度控制
  3. 批处理函数的批量维度保持
  4. 数据类型边界（float32/complex64到float64/complex128）
  5. fft_length参数的裁剪/填充行为

- 可选路径（中/低优先级合并为一组列表）：
  - 高维变换（fft2d/fft3d）的维度正确性
  - 不同设备（CPU/GPU）的一致性
  - 执行模式（eager/graph）的等价性
  - 大尺寸张量的内存处理
  - 复数输入的相位保持
  - 实数输入的对称性验证

- 已知风险/缺失信息（仅列条目，不展开）：
  - batch_*函数文档缺失（仅"TODO: add doc"）
  - IRFFT奇数长度处理细节不明确
  - 形状约束说明不完整
  - 复数类型支持范围有限
  - 梯度计算未在范围内
  - 性能特性未定义