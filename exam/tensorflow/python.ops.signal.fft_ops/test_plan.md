# tensorflow.python.ops.signal.fft_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 参考实现：NumPy FFT 函数作为 oracle
- 测试级别：单元测试，验证功能正确性

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03
- DEFERRED_SET: CASE_04, CASE_05
- 测试文件路径：tests/test_tensorflow_python_ops_signal_fft_ops.py
- 断言分级策略：首轮使用 weak 断言，最终轮启用 strong 断言
- 预算策略：每个用例 size=S，max_lines=80，max_params=6
- 迭代策略：
  - 首轮：仅生成 SMOKE_SET，使用 weak 断言，最多 5 个用例
  - 后续轮：仅修复失败用例，限制 3 个用例，提升 deferred 用例
  - 最终轮：启用 strong 断言，可选覆盖率提升

## 3. 数据与边界
- 正常数据集：随机生成符合形状和数据类型要求的张量
- 边界值：
  - 空张量（零维）
  - 单元素张量
  - 偶数/奇数长度张量
  - 小尺寸张量（1-16 元素）
  - 2D/3D 张量形状
- 极端形状：内存边界的大尺寸张量（后续扩展）
- 负例场景：
  - 非 Tensor 输入
  - 无效轴索引
  - 不支持的数据类型
  - fft_length 小于输入长度

## 4. 覆盖映射
| TC ID | 对应需求/约束 | 覆盖函数 |
|-------|--------------|----------|
| TC-01 | fftshift/ifftshift 与 NumPy 等效性 | fftshift |
| TC-02 | fft/ifft 正向和逆向变换互逆性 | fft, ifft |
| TC-03 | rfft/irfft 实数变换正确性 | rfft, irfft |
| TC-04 | 不同轴参数处理正确性 | fftshift, ifftshift |
| TC-05 | 边界情况处理 | 所有函数 |

### 尚未覆盖的风险点
- 梯度计算正确性（特别是 Hermitian 对称性处理）
- 大尺寸张量的内存使用和性能
- 不同设备类型（CPU/GPU/TPU）的兼容性
- 部分函数通过 gen_spectral_ops 实现，源码可见性有限
- 实数 FFT 返回分量数目的数学依据验证

## 5. 依赖与 Mock 策略
- 主要依赖：NumPy（作为 oracle 参考）
- 需要 mock 的目标（根据 requirements）：
  - tensorflow.python.ops.manip_ops.roll
  - tensorflow.python.ops.gen_spectral_ops
  - tensorflow.python.ops.math_ops
  - tensorflow.python.framework.ops
- 首轮用例：无需 mock，直接测试真实实现
- 后续用例：根据测试需要选择性 mock 底层依赖