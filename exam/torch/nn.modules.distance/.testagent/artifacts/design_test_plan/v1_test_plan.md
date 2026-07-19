# torch.nn.modules.distance 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：直接测试模块类，无需mock（底层functional函数已稳定）
- 随机性处理：固定随机种子，控制张量生成
- 设备策略：首轮仅CPU，后续扩展CUDA

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04（4个核心用例）
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07, CASE_08（4个扩展用例）
- **group 列表**: G1(PairwiseDistance), G2(CosineSimilarity)
- **active_group_order**: G1 → G2（先距离后相似度）
- **断言分级策略**: 首轮仅weak断言（shape/dtype/finite/basic_property）
- **预算策略**: 
  - S级用例: max_lines=60, max_params=6
  - M级用例: max_lines=80, max_params=8
  - 首轮总用例数: 4个

## 3. 数据与边界
- **正常数据集**: 随机正态分布张量，固定种子确保可复现
- **边界值**: 
  - p值边界: 1.0, 2.0, -1.0, inf（曼哈顿/欧氏/负范数）
  - eps边界: 1e-8, 1e-4, 1e-6（极小/中等/默认）
  - 形状边界: 小(2,3), 中(3,4), 大(5,6)
  - 维度边界: dim=0,1,2（多维度输入）
- **极端形状**: 空张量、零向量、单元素张量
- **负例场景**: 
  - 形状不匹配异常
  - 维度越界异常
  - 非法参数（负eps）
  - 类型错误输入

## 4. 覆盖映射
| TC_ID | 需求覆盖 | 约束覆盖 | 风险点 |
|-------|----------|----------|--------|
| TC-01 | PairwiseDistance默认参数 | p=2.0, eps=1e-6, keepdim=False | 欧氏距离计算正确性 |
| TC-02 | PairwiseDistance参数边界 | p=1.0/2.0, eps=1e-8/1e-4, keepdim=True/False | 参数组合行为 |
| TC-03 | CosineSimilarity默认参数 | dim=1, eps=1e-8 | 余弦相似度计算正确性 |
| TC-04 | CosineSimilarity不同维度 | dim=0/2, 多维度输入 | 维度选择效果 |
| TC-05 | 异常输入处理 | 形状不匹配异常 | 错误处理完整性 |
| TC-06 | 负p值行为 | p=-1.0 | 负范数未明确行为 |
| TC-07 | 零向量处理 | 全零输入 | 数值稳定性 |
| TC-08 | 维度异常 | dim越界 | 边界条件检查 |

**尚未覆盖的关键风险点**:
- p为inf时的极限行为
- 输入包含inf/nan值的处理
- 广播机制的边界条件
- 内存使用和性能基准
- 梯度计算正确性（如需）