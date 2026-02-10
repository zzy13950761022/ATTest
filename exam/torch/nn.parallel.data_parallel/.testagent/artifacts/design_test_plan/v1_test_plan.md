# torch.nn.parallel.data_parallel 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（针对scatter/gather操作）
- 随机性处理：固定随机种子，控制RNG状态
- 设备管理：CUDA设备可用性检查，设备状态隔离

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07
- **group列表**: G1（核心执行路径）, G2（参数与边界处理）
- **active_group_order**: G1 → G2
- **断言分级策略**: 首轮使用weak断言，最终轮启用strong断言
- **预算策略**: 
  - size: S/M（小型/中型测试）
  - max_lines: 70-85行
  - max_params: 5-6个参数

## 3. 数据与边界
- **正常数据集**: 随机生成符合模块输入要求的张量
- **边界值**: 
  - 空device_ids列表（使用所有GPU）
  - 单GPU设备执行
  - CPU输出设备（output_device=-1）
  - 小形状输入（[1, 10]）
  - 不同dim值（0, 1）
- **极端形状**: 大batch size，高维输入
- **负例与异常场景**:
  - 无效module类型
  - 无效inputs类型
  - 无效GPU设备ID
  - 模块参数不在正确设备
  - dim超出范围
  - 无效module_kwargs类型

## 4. 覆盖映射
| TC ID | 需求/约束覆盖 | 优先级 |
|-------|---------------|--------|
| TC-01 | 单GPU设备正常执行 | High |
| TC-02 | 多GPU设备并行执行 | High |
| TC-03 | CPU作为输出设备 | High |
| TC-04 | 带module_kwargs传递 | High |

**尚未覆盖的风险点**:
- 内存不足场景处理
- 混合精度输入测试
- 嵌套模块结构测试
- 梯度计算兼容性验证
- 极端大形状输入的内存限制