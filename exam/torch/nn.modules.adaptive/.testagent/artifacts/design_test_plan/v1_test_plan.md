# torch.nn.modules.adaptive 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 pytest fixtures 管理测试资源
- 随机性处理：固定随机种子确保测试可重复性
- 设备隔离：分别测试 CPU 和 CUDA（如果可用）环境

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07, CASE_08
- **group 列表**: 
  - G1: AdaptiveLogSoftmaxWithLoss 核心功能
  - G2: 参数验证与辅助方法
- **active_group_order**: G1, G2
- **断言分级策略**: 首轮使用 weak 断言，最终轮启用 strong 断言
- **预算策略**: 
  - size: S（小型测试）
  - max_lines: 75-90 行
  - max_params: 8 个参数

## 3. 数据与边界
- **正常数据集**: 随机生成符合标签频率排序的输入
- **边界值**: 
  - 最小 n_classes=2
  - 单元素 cutoffs 列表
  - 极端 div_value 值（接近 0 或极大）
- **极端形状**: 
  - 空批处理 (0, in_features)
  - 大 in_features (1000+)
  - 大 n_classes (10000+)
- **负例与异常场景**:
  - cutoffs 非递增序列
  - cutoffs 包含重复值
  - cutoffs 值超出范围
  - n_classes < 2
  - in_features ≤ 0
  - 输入/目标形状不匹配
  - 目标值超出有效范围

## 4. 覆盖映射
| TC_ID | 需求/约束覆盖 | 风险点 |
|-------|--------------|--------|
| TC-01 | 基本前向传播功能 | 数值稳定性 |
| TC-02 | 批处理兼容性 | 形状处理逻辑 |
| TC-03 | 参数验证 | 异常处理完整性 |
| TC-04 | 辅助方法正确性 | log_prob 数值范围 |
| TC-05 | 设备兼容性 | CUDA 可用性依赖 |

**尚未覆盖的关键风险点**:
- 标签频率排序的实际验证方法
- 大规模 n_classes 的性能退化
- 混合精度训练兼容性
- 梯度计算正确性验证
- 内存使用效率测试

## 5. 迭代策略
- **首轮 (round1)**: 仅生成 SMOKE_SET 用例，使用 weak 断言
- **中间轮 (roundN)**: 修复失败用例，提升 deferred 用例
- **最终轮 (final)**: 启用 strong 断言，可选覆盖率检查