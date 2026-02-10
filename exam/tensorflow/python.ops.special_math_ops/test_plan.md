# tensorflow.python.ops.special_math_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 断言分级：weak（基础验证）→ strong（数值精度验证）
- 测试文件：单文件集中测试所有功能

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04, CASE_05
- DEFERRED_SET: 无（首轮全覆盖核心功能）
- 测试文件路径：tests/test_tensorflow_python_ops_special_math_ops.py
- 断言分级策略：首轮使用 weak 断言，最终轮启用 strong 断言
- 预算策略：size=S, max_lines=80, max_params=6

## 3. 数据与边界
- 正常数据集：随机正态分布、均匀分布、正数分布
- 边界值：零值、大值、特殊点（0, ±inf, NaN）
- 极端形状：高维张量（>5维）、空维度
- 负例场景：无效数据类型、维度不匹配、定义域外输入
- 异常场景：空方程、无效优化策略、SparseTensor 兼容性

## 4. 覆盖映射
| TC ID | 功能覆盖 | 需求/约束 | 优先级 |
|-------|----------|-----------|--------|
| TC-01 | 贝塞尔函数基本功能 | 特殊函数计算、数据类型支持 | High |
| TC-02 | einsum核心操作 | 张量收缩运算、方程语法 | High |
| TC-03 | 特殊函数边界值 | 边界值处理、数值稳定性 | High |
| TC-04 | 数据类型兼容性 | float32/float64/half 支持 | High |
| TC-05 | lbeta基本功能 | 对数 Beta 函数、维度缩减 | High |

## 5. 尚未覆盖的风险点
- SparseTensor 支持验证
- einsum 优化策略效果对比
- 高维张量（>5维）的特殊函数计算
- 与 SciPy 实现的数值一致性详细验证
- 梯度计算和自动微分验证
- 混合精度计算转换

## 6. 迭代策略
- 首轮（round1）：生成 SMOKE_SET 的 5 个核心用例，使用 weak 断言
- 中间轮（roundN）：修复失败用例，提升 deferred 用例，限制 3 个块
- 最终轮（final）：启用 strong 断言，可选覆盖率提升

## 7. Mock 目标
- 无首轮 mock 需求（直接测试真实实现）
- 后续可能需要 mock：gen_special_math_ops, gen_linalg_ops, math_ops, array_ops, opt_einsum.contract

## 8. 参考实现
- 贝塞尔函数：scipy.special (i0, i1, j0, j1, y0, y1, k0, k1)
- 特殊函数：scipy.special (dawsn, fresnel, expint, spence)
- einsum：numpy.einsum
- lbeta：scipy.special.betaln