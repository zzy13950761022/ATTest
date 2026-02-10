# tensorflow.python.ops.signal.spectral_ops 测试需求

## 1. 目标与范围
- 验证STFT/MDCT正向/逆向变换的正确性和可逆性
- 确保TPU/GPU兼容性和梯度计算支持
- 验证窗口函数约束和完美重构条件
- 不在范围内：性能基准测试、内存使用分析、第三方窗口函数验证

## 2. 输入与约束
- **stft**: signals([..., samples], float32/64), frame_length(int), frame_step(int), fft_length(int|None), window_fn(callable), pad_end(bool)
- **inverse_stft**: stfts([..., frames, fft_unique_bins], complex64/128), frame_length(int), frame_step(int), fft_length(int|None), window_fn(callable)
- **mdct**: signals([..., samples], float32/64), frame_length(int, 必须被4整除), window_fn(callable), pad_end(bool), norm("ortho"|None)
- **inverse_mdct**: mdcts([..., frames, frame_length//2], float32/64), window_fn(callable), norm("ortho"|None)
- 必需组合：frame_length > 0, frame_step > 0, fft_length >= frame_length
- 随机性要求：无全局状态依赖，确定性计算

## 3. 输出与判定
- stft返回：[..., frames, fft_unique_bins] complex64/128张量，fft_unique_bins = fft_length//2 + 1
- inverse_stft返回：[..., samples] float32/64信号张量，长度 = (frames-1)*frame_step + frame_length
- mdct返回：[..., frames, frame_length//2] float32/64张量
- inverse_mdct返回：[..., samples] float32/64信号张量
- 容差：浮点误差<1e-6，复数相位误差<1e-8
- 副作用检查：无外部状态修改，无文件/网络操作

## 4. 错误与异常场景
- 非法输入：frame_length <= 0, frame_step <= 0, fft_length < frame_length
- 类型错误：非数值张量，非整数参数，非可调用window_fn
- 维度错误：signals维度<1，stfts维度<2，mdcts维度<2
- 边界值：空张量，零长度信号，frame_length=1，极端形状(1x1, 1000x1000)
- 数值边界：NaN, Inf, 极大/极小浮点数
- mdct专用：frame_length不能被4整除，无效norm值

## 5. 依赖与环境
- 外部依赖：无网络/文件/数据库依赖
- 需要mock：无外部服务调用
- 需要monkeypatch：无
- 设备要求：支持CPU/GPU/TPU测试环境
- 资源依赖：tensorflow运行时，numpy用于参考实现验证

## 6. 覆盖与优先级
- **必测路径（高优先级）**：
  1. STFT正向+逆向变换的完美重构验证
  2. MDCT正向+逆向变换的完美重构验证（frame_length能被4整除）
  3. 不同数据类型组合：float32/complex64, float64/complex128
  4. 边界情况：信号长度小于窗口长度，pad_end=True/False
  5. 梯度计算验证（所有可微参数）

- **可选路径（中/低优先级）**：
  - 自定义窗口函数验证
  - 不同fft_length值的影响
  - 多维输入张量处理
  - 批量处理性能
  - 与numpy/scipy参考实现的对比
  - 内存使用和形状推断

- **已知风险/缺失信息**：
  - 窗口函数的数学约束条件验证
  - 极端数值稳定性测试
  - TPU特定行为差异
  - 并发/并行执行安全性
  - 版本兼容性（不同TF版本）