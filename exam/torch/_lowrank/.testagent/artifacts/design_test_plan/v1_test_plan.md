# torch._lowrank 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：固定随机种子确保可重复性
- 随机性处理：使用 torch.manual_seed 控制 RNG
- 设备支持：优先 CPU，GPU 作为扩展
- 数据类型：float32 为主，float64 为扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03（3个核心用例）
- **DEFERRED_SET**: CASE_04, CASE_05, CASE_06（3个延后用例）
- **group 列表**:
  - G1: 核心函数族 - svd_lowrank（2个smoke，1个deferred）
  - G2: 辅助函数族 - get_approximate_basis 和 pca_lowrank（1个smoke，2个deferred）
- **active_group_order**: G1, G2（按优先级顺序）
- **断言分级策略**: 首轮使用 weak 断言（形状、类型、有限性、基本属性）
- **预算策略**: 
  - 每个用例最大 80 行代码
  - 最多 6 个参数
  - 所有用例均为参数化测试
  - 无需 mock（首轮）

## 3. 数据与边界
- **正常数据集**: 随机生成的低秩矩阵，形状适中（5×3, 6×4, 4×4）
- **随机生成策略**: 使用 torch.randn 生成正态分布数据
- **边界值**:
  - q = 0（零秩）和 q = min(m,n)（满秩）
  - niter = 0（无迭代）和 niter = 2（默认）
  - 最小形状 (1,1) 和极端形状 (1000,10)
- **极端形状**: 高矩阵 (m≫n) 和宽矩阵 (n≫m)
- **空输入**: 不支持空张量（需异常测试）
- **负例与异常场景**:
  - q > min(m,n) 引发 ValueError
  - q < 0 引发 ValueError
  - niter < 0 引发 ValueError
  - 非 Tensor 输入引发 TypeError
  - M 形状不匹配引发 RuntimeError

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖约束 | 优先级 |
|-------|----------|----------|--------|
| TC-01 | 验证 svd_lowrank 基本功能 | 形状验证、类型检查、基本属性 | High |
| TC-02 | 测试边界条件（满秩） | q = min(m,n) 场景验证 | High |
| TC-03 | 验证 get_approximate_basis | 正交基生成、形状正确性 | High |
| TC-04 | 随机性控制验证 | 种子固定确保可重复性 | Medium |
| TC-05 | pca_lowrank 中心化测试 | center=True/False 对比 | Medium |
| TC-06 | 稀疏矩阵支持验证 | 仅 pca_lowrank 支持稀疏 | Medium |

## 5. 尚未覆盖的风险点
- GPU 设备兼容性测试（延后）
- 不同 dtype 的数值精度验证
- 病态矩阵的数值稳定性
- 批量处理 (*, m, n) 形状的全面验证
- 与完整 SVD 结果的详细对比分析
- M 参数（均值张量）的影响测试