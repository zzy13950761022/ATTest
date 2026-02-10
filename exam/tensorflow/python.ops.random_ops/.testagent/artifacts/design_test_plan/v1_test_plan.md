# tensorflow.python.ops.random_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：固定随机种子控制随机性，使用fixtures管理测试资源
- 随机性处理：固定随机种子确保可重复性，统计检验验证分布特性
- 设备支持：优先CPU测试，GPU作为可选扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (random_normal基本功能), CASE_02 (random_uniform基本功能), CASE_03 (truncated_normal截断特性)
- **DEFERRED_SET**: CASE_04 (random_gamma分布验证), CASE_05 (random_poisson_v2泊松分布)
- **测试文件路径**: tests/test_tensorflow_python_ops_random_ops.py (单文件)
- **断言分级策略**: 首轮使用weak断言（形状、数据类型、基本属性），后续启用strong断言（统计检验、分布特性）
- **预算策略**: 每个用例size=S，max_lines=80，max_params=6，全部参数化

## 3. 数据与边界
- **正常数据集**: 标准形状[10], [5,5], [100]等，常用参数范围
- **随机生成策略**: 固定种子确保可重复性，大样本用于统计检验
- **边界值**: 空张量shape=[0]，极小/极大参数值，极端形状
- **负例与异常场景**:
  - 非法shape（负数、非整数）
  - 无效参数（stddev<=0, minval>=maxval）
  - 不支持的数据类型
  - 广播失败的不兼容形状

## 4. 覆盖映射
| TC ID | 对应需求 | 核心功能验证 |
|-------|----------|--------------|
| TC-01 | 基本形状和dtype正确性 | random_normal正态分布 |
| TC-02 | 种子机制可重复性 | random_uniform均匀分布 |
| TC-03 | 统计分布特性 | truncated_normal截断特性 |
| TC-04 | 参数广播规则 | random_gamma伽马分布 |
| TC-05 | 边界值处理 | random_poisson_v2泊松分布 |

## 5. 尚未覆盖的风险点
- 整数均匀分布的偏差问题（maxval-minval非2的幂时）
- 伽马分布alpha<<1的数值稳定性
- 不同硬件后端（CPU/GPU）的浮点精度差异
- 全局种子与操作种子的复杂交互
- 截断正态分布的拒绝采样效率问题

## 6. 迭代策略
- **首轮**: 仅生成SMOKE_SET用例，使用weak断言，验证核心功能
- **后续轮次**: 修复失败用例，逐步启用DEFERRED_SET，添加参数扩展
- **最终轮次**: 启用strong断言，进行统计检验，覆盖边界情况