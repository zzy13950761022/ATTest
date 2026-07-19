# torch.ao.nn.quantized.functional 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_05, CASE_08
- DEFERRED_SET: CASE_03, CASE_04, CASE_06, CASE_07, CASE_09, CASE_10
- group 列表与 active_group_order: G1, G2, G3
- 断言分级策略：首轮使用weak断言，最终轮启用strong断言
- 预算策略：每个CASE限制80行代码，最多8个参数

## 3. 数据与边界
- 正常数据集：标准形状量化张量，合理量化参数
- 随机生成策略：固定种子生成随机量化张量
- 边界值：最小形状(1,1,3,3)，极端量化参数(scale=0.1, zero_point=255)
- 极端形状：不同batch size和通道数组合
- 空输入：不测试空张量（量化张量不支持）
- 负例与异常场景：
  - 非量化张量输入
  - 错误的dtype组合
  - 无效的padding_mode
  - groups不能整除in_channels
  - scale <= 0的量化参数

## 4. 覆盖映射
- TC-01: 基本量化卷积操作正确性验证
- TC-02: 量化参数正确传播到输出
- TC-05: 线性层量化操作正确性
- TC-08: 激活函数量化操作正确性
- 尚未覆盖的风险点：
  - 设备兼容性（CPU/GPU）
  - 已弃用函数（upsample系列）
  - 某些量化数据类型组合
  - 性能基准测试（linear函数权重打包开销）