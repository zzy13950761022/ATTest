# tensorflow.python.ops.gen_spectral_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（针对底层执行和操作定义）
- 随机性处理：固定随机种子，使用确定性数据生成
- 设备隔离：CPU/GPU设备一致性验证
- 执行模式：支持eager和graph模式等价性

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基本FFT变换正确性）、CASE_02（RFFT实数变换与长度控制）、CASE_03（批处理FFT维度保持）
- **DEFERRED_SET**: CASE_04（数据类型边界验证）、CASE_05（fft_length裁剪填充行为）
- **测试文件路径**: tests/test_tensorflow_python_ops_gen_spectral_ops.py（单文件）
- **断言分级策略**: 首轮使用weak断言（shape/dtype/finite/basic_property），后续启用strong断言
- **预算策略**: 
  - 每个CASE: size=S, max_lines=80-85, max_params=6-7
  - 参数化测试：支持多维度扩展
  - Mock需求：仅CASE_03需要mock底层执行

## 3. 数据与边界
- **正常数据集**: 随机复数/实数张量，形状[8]/[16]/[4,8]，固定随机种子
- **边界值处理**:
  - 最小有效fft_length（1）
  - 奇数长度IRFFT特殊处理
  - 长度裁剪（输入>fft_length）
  - 长度填充（输入<fft_length）
- **极端形状**: 空张量、零长度维度、大尺寸内存测试
- **数据类型边界**: float32/complex64到float64/complex128
- **负例与异常场景**:
  - 非数值类型输入
  - 不支持的数据类型
  - 维度不足错误
  - 无效fft_length（非正整数）
  - 复数输入给RFFT
  - 极大fft_length内存不足

## 4. 覆盖映射
| TC ID | 对应需求/约束 | 优先级 | 覆盖要点 |
|-------|--------------|--------|----------|
| TC-01 | 基本FFT/IFFT变换正确性 | High | FFT基本功能、可逆性验证 |
| TC-02 | RFFT/IRFFT实数-复数转换 | High | 长度控制、裁剪填充行为 |
| TC-03 | 批处理函数批量维度保持 | High | 批量维度不变性、文档缺失函数 |
| TC-04 | 数据类型边界验证 | High | 精度边界、误差控制 |
| TC-05 | fft_length参数行为 | High | 裁剪填充、长度控制 |

**尚未覆盖的风险点**:
- IRFFT奇数长度处理细节不明确
- 高维变换（fft2d/fft3d）的维度正确性
- 执行模式（eager/graph）的等价性
- 大尺寸张量的内存处理边界
- 复数输入的相位保持特性
- 实数输入的对称性验证

## 5. 迭代策略
- **首轮（round1）**: 仅生成SMOKE_SET（3个核心用例），使用weak断言
- **后续迭代（roundN）**: 修复失败用例，逐步启用DEFERRED_SET，参数扩展
- **最终轮（final）**: 启用strong断言，可选覆盖率提升，完成所有参数扩展