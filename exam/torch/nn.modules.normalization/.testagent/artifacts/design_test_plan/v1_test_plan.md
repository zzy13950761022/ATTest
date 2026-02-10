# torch.nn.modules.normalization 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：每个测试用例独立实例化归一化层
- 随机性处理：固定随机种子，使用 torch.manual_seed
- 设备隔离：CPU测试为主，CUDA测试作为参数扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07, CASE_08, CASE_09, CASE_10
- **group 列表**: G1(GroupNorm), G2(LayerNorm), G3(LocalResponseNorm/CrossMapLRN2d)
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮使用weak断言（形状、数据类型、有限性、基本属性）
- **预算策略**: 每个CASE最多80行代码，最多6个参数，优先参数化测试

## 3. 数据与边界
- **正常数据集**: 随机生成正态分布张量，形状符合各归一化层要求
- **边界值**: eps极小值(1e-7)、批量大小=1、极端alpha/beta值
- **极端形状**: 大尺寸输入(64x64)、小尺寸输入(4x4)、不同维度(2D/3D/4D)
- **空输入**: 不适用（归一化层需要有效输入）
- **负例与异常场景**:
  - GroupNorm整除性异常
  - 非法normalized_shape形状
  - 无效参数值（负数size、零eps）
  - 设备/数据类型不匹配

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖约束 |
|-------|----------|----------|
| TC-01 | GroupNorm基本功能 | 高优先级必测路径 |
| TC-02 | GroupNorm异常处理 | 整除性检查异常 |
| TC-03 | LayerNorm基本功能 | 不同normalized_shape支持 |
| TC-04 | LocalResponseNorm基本功能 | 跨通道归一化正确性 |

**尚未覆盖的风险点**:
- CrossMapLRN2d与LocalResponseNorm差异对比
- 训练/评估模式一致性验证
- 极端数值稳定性（极大/极小值）
- 不同维度输入全面支持（2D/3D/4D）