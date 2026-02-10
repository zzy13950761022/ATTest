# torch.autograd.functional 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用fixtures管理测试数据，mock向量化内部调用
- 随机性处理：固定随机种子确保结果可重现
- 设备管理：支持CPU和CUDA设备测试

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (vjp基本功能), CASE_02 (jacobian矩阵), CASE_03 (vhp基本功能), CASE_04 (create_graph参数)
- **DEFERRED_SET**: CASE_05 (strict模式检测), CASE_06 (向量化功能)
- **group列表**: G1 (核心微分函数族), G2 (高阶微分与向量化)
- **active_group_order**: G1 → G2
- **断言分级策略**: 首轮使用weak断言（形状/类型/有限值/基本属性），后续启用strong断言（数值对比/性能检查）
- **预算策略**: 
  - S级用例: max_lines=80, max_params=6
  - M级用例: max_lines=100, max_params=8
  - 参数化用例优先，非参数化用例作为补充

## 3. 数据与边界
- **正常数据集**: 随机生成浮点张量，形状[2,2]到[4,4]，dtype float32/float64
- **边界值处理**:
  - 空Tensor（形状含0维度）
  - 标量输入（0维Tensor）
  - 极大/极小数值（inf, nan, 极值）
  - 大维度张量（内存边界测试）
  - 嵌套元组深度边界
- **负例与异常场景**:
  - 非Tensor输入触发TypeError
  - 函数返回非Tensor触发RuntimeError
  - 形状不匹配触发RuntimeError
  - 无效策略参数触发ValueError
  - 不兼容参数组合触发RuntimeError

## 4. 覆盖映射
| TC_ID | 需求覆盖 | 约束覆盖 | 风险点 |
|-------|----------|----------|--------|
| TC-01 | 基本vjp功能验证 | 输入必须是Tensor | 梯度计算正确性 |
| TC-02 | jacobian矩阵正确性 | 支持reverse-mode策略 | 数值稳定性 |
| TC-03 | vhp基本功能验证 | 向量-Hessian积计算 | Hessian对称性 |
| TC-04 | create_graph参数 | 计算图创建控制 | 内存泄漏风险 |
| TC-05 | strict模式检测 | 独立输入输出检测 | 性能影响 |
| TC-06 | 向量化功能 | vectorize实验性功能 | 兼容性问题 |

**尚未覆盖的关键风险点**:
- strict=True与vectorize=True不兼容
- create_graph=True与正向模式不兼容
- 复杂嵌套函数链式微分
- 混合精度计算边界情况
- 多设备一致性验证