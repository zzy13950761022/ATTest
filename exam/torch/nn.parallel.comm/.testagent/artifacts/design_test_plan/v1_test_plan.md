# torch.nn.parallel.comm 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（GPU设备模拟、底层C++调用）
- 随机性处理：固定随机种子生成测试数据
- 设备依赖：至少需要2个可用GPU设备，支持CUDA环境

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (broadcast基本功能), CASE_02 (broadcast参数冲突异常), CASE_05 (reduce_add基本归约), CASE_08 (scatter_gather往返完整性)
- **DEFERRED_SET**: CASE_03, CASE_04, CASE_06, CASE_07, CASE_09, CASE_10
- **group列表**: G1(广播函数族), G2(归约函数族), G3(分散聚集函数族)
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮使用weak断言（形状、数据类型、设备、基本数据正确性），后续启用strong断言（近似相等、梯度检查、性能验证）
- **预算策略**: 
  - S级用例: max_lines ≤ 80, max_params ≤ 6
  - M级用例: max_lines ≤ 90, max_params ≤ 6
  - 参数化用例优先，减少重复代码

## 3. 数据与边界
- **正常数据集**: 小到中等形状张量（[2,3]到[10,10]），float32/float64数据类型，2-3个GPU设备
- **随机生成策略**: 固定种子生成随机张量，确保测试可重复
- **边界值**:
  - 空设备列表（异常场景）
  - 零维张量（边界形状）
  - 极端大形状（内存边界测试）
  - 复数张量（特殊数据类型）
  - 缓冲区大小边界（1字节到10MB）
- **负例与异常场景**:
  1. 参数组合冲突（devices/out同时提供）
  2. 设备不一致（broadcast_coalesced要求）
  3. 形状不匹配（reduce_add要求）
  4. 非GPU张量输入（reduce_add要求）
  5. 维度超出范围
  6. chunk_sizes总和与输入维度不匹配

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖约束 | 优先级 |
|-------|----------|----------|--------|
| TC-01 | 单GPU到多GPU广播功能正确性 | devices/out二选一，数据一致性 | High |
| TC-02 | 参数组合冲突的异常处理 | 参数冲突验证 | High |
| TC-03 | 合并广播功能 | 缓冲区大小影响 | Medium |
| TC-04 | 设备一致性检查 | 同一设备要求 | Medium |
| TC-05 | 多GPU张量归约求和数值正确性 | 形状匹配，GPU要求 | High |
| TC-06 | 形状不匹配异常 | 输入验证 | Medium |
| TC-07 | 合并归约功能 | 嵌套张量处理 | Medium |
| TC-08 | 张量分散-聚集往返数据完整性 | 参数互斥，数据完整性 | High |
| TC-09 | scatter参数冲突 | 参数验证 | Medium |
| TC-10 | gather参数互斥 | 参数验证 | Medium |

**尚未覆盖的风险点**:
- 底层C++实现细节未知
- 流同步行为未明确文档化
- 混合精度支持验证
- 稀疏张量处理
- 大规模张量内存管理
- 跨设备类型兼容性（CPU/GPU混合）
- NCCL后端特定行为