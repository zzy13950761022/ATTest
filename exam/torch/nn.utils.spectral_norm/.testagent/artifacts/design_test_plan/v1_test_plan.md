# torch.nn.utils.spectral_norm 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：直接测试函数，无需mock（首轮）
- 随机性处理：固定随机种子确保幂迭代可重复性
- 设备支持：首轮仅测试CPU，后续扩展GPU

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_05, CASE_09（4个核心用例）
- **DEFERRED_SET**: CASE_03, CASE_04, CASE_06, CASE_07, CASE_08, CASE_10, CASE_11, CASE_12
- **group列表**: G1（核心功能）, G2（参数处理）, G3（异常处理）
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮仅使用weak断言，最终轮启用strong断言
- **预算策略**: 每个用例max_lines≤80, max_params≤7, size=S

## 3. 数据与边界
- **正常数据集**: 标准Linear/Conv层，中等尺寸权重矩阵
- **随机生成策略**: 固定随机种子，可重复的权重初始化
- **边界值测试**:
  - 零权重矩阵（全零）
  - 单位矩阵权重
  - 极小eps值（1e-30）
  - 零迭代次数（n_power_iterations=0）
  - 1×1极小形状权重
- **负例与异常场景**:
  - 参数不存在（AttributeError）
  - 负迭代次数（ValueError）
  - 非正eps值（ValueError）
  - 无效dim索引（IndexError）
  - 一维权重张量（ValueError）

## 4. 覆盖映射
| TC ID | 对应需求/约束 | 优先级 |
|-------|--------------|--------|
| TC-01 | 标准线性层谱归一化 | High |
| TC-02 | ConvTranspose特殊dim处理 | High |
| TC-03 | 自定义参数名支持 | High |
| TC-04 | 多轮幂迭代验证 | High |
| TC-05 | 不同模块类型兼容性 | High |
| TC-09 | 参数不存在异常处理 | High |

**尚未覆盖的风险点**:
- 函数已标记为弃用的兼容性影响
- 超大权重矩阵的内存和性能问题
- GPU设备上的计算差异
- 与其他PyTorch功能的集成问题
- 前向传播钩子的线程安全性