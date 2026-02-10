# torch.nn.modules.fold 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：每个测试独立实例化Fold/Unfold对象
- 随机性处理：固定随机种子，使用torch.manual_seed
- 设备支持：首轮仅测试CPU，后续扩展CUDA
- 数据类型：首轮仅测试float32，后续扩展float64

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_05, CASE_06, CASE_09
- **DEFERRED_SET**: CASE_03, CASE_04, CASE_07, CASE_08, CASE_10, CASE_11
- **group列表**: G1(Fold类), G2(Unfold类), G3(组合测试)
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮仅使用weak断言，最终轮启用strong断言
- **预算策略**: 
  - S级用例: max_lines=60, max_params=8-11
  - M级用例: max_lines=80, max_params=9
  - 所有用例均为参数化测试

## 3. 数据与边界
- **正常数据集**: 随机生成符合形状约束的张量
- **边界值**: 最小有效尺寸(2x2), 单通道, 单批次
- **极端形状**: 大尺寸输入(100x100), 多通道(64), 大批量(32)
- **空输入**: 不支持空张量，测试零值参数异常
- **负例场景**: 
  - kernel_size=0或负值
  - output_size小于kernel_size
  - 不支持的张量维度(1D, 5D+)
  - 形状不满足数学公式约束

## 4. 覆盖映射
| TC ID | 需求覆盖 | 约束覆盖 |
|-------|----------|----------|
| TC-01 | Fold基本功能(int参数) | 参数类型处理 |
| TC-02 | Fold基本功能(tuple参数) | 参数类型处理 |
| TC-03 | Fold边界条件 | 最小有效输入 |
| TC-04 | Fold错误处理 | 形状约束验证 |
| TC-05 | Unfold基本功能(int参数) | 参数类型处理 |
| TC-06 | Unfold基本功能(tuple参数) | 参数类型处理 |
| TC-07 | Unfold边界条件 | 最小有效输入 |
| TC-08 | Unfold错误处理 | 参数有效性验证 |
| TC-09 | Fold-Unfold组合基本 | 数学一致性 |
| TC-10 | Fold-Unfold组合重叠 | 重叠块处理 |
| TC-11 | Fold-Unfold组合dilation | dilation参数处理 |

## 5. 尚未覆盖的风险点
- 具体支持的dtype范围未完全验证
- 超大张量内存处理边界
- CUDA设备兼容性
- 非标准形状的公式约束验证
- 错误消息格式和类型一致性