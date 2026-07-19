# tensorflow.python.ops.signal.spectral_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用tensorflow测试工具，固定随机种子
- 随机性处理：使用确定性随机数生成器，固定随机种子
- 设备支持：优先CPU测试，支持GPU/TPU环境检测

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (STFT正向变换), CASE_03 (STFT完美重构), CASE_04 (MDCT正向变换)
- **DEFERRED_SET**: CASE_02 (STFT逆向变换), CASE_05 (MDCT完美重构)
- **测试文件路径**: tests/test_tensorflow_python_ops_signal_spectral_ops.py
- **断言分级策略**: 首轮使用weak断言，最终轮启用strong断言
- **预算策略**: 
  - S级用例: max_lines=80, max_params=8
  - M级用例: max_lines=100, max_params=8
  - 所有用例均为参数化测试

## 3. 数据与边界
- **正常数据集**: 随机生成的正态分布信号，形状[100-200]
- **边界值**: 信号长度小于窗口长度，frame_length=1，极端形状(1x1)
- **空输入**: 零长度信号张量，空维度
- **负例场景**: 
  - frame_length <= 0, frame_step <= 0
  - fft_length < frame_length
  - frame_length不能被4整除（MDCT）
  - 无效norm值
  - 非数值张量输入
- **数值边界**: NaN, Inf, 极大/极小浮点数

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖函数 | 关键约束 |
|-------|----------|----------|----------|
| TC-01 | STFT正向变换 | stft | 数据类型兼容，形状推断 |
| TC-02 | STFT逆向变换 | inverse_stft | 复数输入处理，重构长度 |
| TC-03 | STFT完美重构 | stft+inverse_stft | 可逆性验证，窗口函数约束 |
| TC-04 | MDCT正向变换 | mdct | frame_length整除4，正交性 |
| TC-05 | MDCT完美重构 | mdct+inverse_mdct | 完美重构，归一化选项 |

## 5. 尚未覆盖的风险点
- 窗口函数的数学约束条件验证
- 梯度计算正确性（所有可微参数）
- TPU特定行为差异测试
- 并发/并行执行安全性
- 自定义窗口函数兼容性
- 极端数值稳定性（下溢/上溢）

## 6. 迭代策略
1. **首轮**: 仅生成SMOKE_SET用例，使用weak断言
2. **后续轮**: 修复失败用例，从DEFERRED_SET提升用例
3. **最终轮**: 启用strong断言，可选覆盖率目标

## 7. 参考实现
- STFT: numpy.fft.rfft / numpy.fft.irfft
- MDCT: scipy.fft.dct (DCT-IV变换)
- 自一致性验证: 正向+逆向变换的完美重构