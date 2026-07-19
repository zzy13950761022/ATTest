# torch.nn.modules.pixelshuffle 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：每个测试用例独立实例化PixelShuffle/PixelUnshuffle
- 随机性处理：固定随机种子，使用torch.manual_seed
- 设备隔离：CPU测试为主，CUDA测试作为扩展

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_05, CASE_06（4个核心用例）
- DEFERRED_SET: CASE_03, CASE_04, CASE_07, CASE_08（4个扩展用例）
- group列表：G1(PixelShuffle核心功能), G2(PixelUnshuffle与互逆验证)
- active_group_order: G1 → G2
- 断言分级策略：首轮使用weak断言（形状、数据类型、有限值）
- 预算策略：所有用例size=S，max_lines≤70，max_params≤4

## 3. 数据与边界
- 正常数据集：随机生成符合整除条件的张量
- 边界值：缩放因子=1（恒等变换），大缩放因子（3,4）
- 极端形状：0批次维度，不同批次大小（1,2,4）
- 空输入：不支持（需要至少4维张量）
- 负例场景：非整数缩放因子，不满足整除条件，维度不足

## 4. 覆盖映射
| TC_ID | 需求覆盖 | 约束验证 |
|-------|----------|----------|
| TC-01 | PixelShuffle基本功能 | 形状变换公式，数据类型保持 |
| TC-02 | 缩放因子边界 | 缩放因子=1的恒等性 |
| TC-03 | 不同批次大小 | 0批次维度处理 |
| TC-04 | 不同数据类型 | float32/float64支持 |
| TC-05 | PixelUnshuffle基本功能 | 逆变换形状公式 |
| TC-06 | 互逆操作验证 | PixelShuffle∘PixelUnshuffle=identity |
| TC-07 | 不同设备支持 | CPU/CUDA一致性 |
| TC-08 | 整除边界条件 | 高度/宽度整除验证 |

## 5. 尚未覆盖的风险点
- 非整数缩放因子异常处理
- 极端大形状内存溢出
- 梯度数值稳定性问题
- 与torch.jit的兼容性
- 自动混合精度支持