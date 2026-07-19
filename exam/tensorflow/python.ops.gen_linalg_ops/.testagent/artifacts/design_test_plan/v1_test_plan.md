# tensorflow.python.ops.gen_linalg_ops 测试计划

## 1. 测试策略
- **单元测试框架**：pytest
- **隔离策略**：mock/monkeypatch/fixtures
  - 核心执行路径：`pywrap_tfe.TFE_Py_FastPathExecute`
  - 梯度记录：`_execute.record_gradient`
  - 错误传播：`_core._NotOkStatusException`
- **随机性处理**：固定随机种子，控制RNG生成确定性测试数据
- **设备兼容性**：优先CPU测试，GPU作为可选扩展
- **数值类型覆盖**：float32/float64优先，half/complex作为后续扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (Cholesky), CASE_02 (QR), CASE_03 (MatrixInverse)
- **DEFERRED_SET**: CASE_04 (SVD), CASE_05 (Batch Operations)
- **测试文件路径**：`tests/test_tensorflow_python_ops_gen_linalg_ops.py`（单文件）
- **断言分级策略**：
  - weak：形状匹配、数据类型、有限值、基本属性
  - strong：数值精度、正交性、重构误差、批量一致性
- **预算策略**：
  - size：S（小型测试，80-90行）
  - max_lines：75-90行
  - max_params：6-7个参数
  - is_parametrized：true（支持参数化扩展）
  - requires_mock：true（需要mock底层执行）

## 3. 数据与边界
- **正常数据集**：随机生成对称正定矩阵、满秩矩阵、良态矩阵
- **随机生成策略**：固定种子，控制条件数，避免病态矩阵
- **边界值**：
  - 最小形状：1x1矩阵
  - 小尺寸：2x2, 3x3
  - 非方阵：3x5, 4x3
  - 批量操作：2x3x3
- **极端形状**：
  - 空张量（后续扩展）
  - 零维张量（后续扩展）
  - 超大矩阵（内存限制，后续扩展）
- **负例与异常场景**：
  - 非正定矩阵调用Cholesky
  - 奇异矩阵求逆
  - 非方阵调用Cholesky/MatrixInverse
  - 不支持的数据类型
  - 形状不匹配
  - NaN/Inf输入
  - 病态矩阵数值稳定性

## 4. 覆盖映射
| TC ID | 对应需求/约束 | 覆盖函数 | 优先级 |
|-------|--------------|----------|--------|
| TC-01 | Cholesky对称正定验证 | gen_linalg_ops.cholesky | High |
| TC-02 | QR分解正交性验证 | gen_linalg_ops.qr | High |
| TC-03 | 矩阵求逆可逆性验证 | gen_linalg_ops.matrix_inverse | High |
| TC-04 | SVD分解重构验证 | gen_linalg_ops.svd | High |
| TC-05 | 批量操作一致性验证 | 多个函数批量支持 | High |

### 尚未覆盖的风险点
1. **复数矩阵操作**：complex64/complex128类型支持
2. **GPU-CPU一致性**：设备间结果差异
3. **梯度计算验证**：前向/反向传播正确性
4. **病态矩阵处理**：数值稳定性边界
5. **文档缺失函数**：banded_triangular_solve等
6. **类型转换规则**：隐式类型转换行为
7. **形状验证细节**：Python层验证规则
8. **不可逆矩阵行为**：错误处理未定义

## 5. 迭代策略
- **首轮（round1）**：仅生成SMOKE_SET（3个核心用例），使用weak断言
- **后续迭代（roundN）**：修复失败用例，提升deferred_set优先级
- **最终轮（final）**：启用strong断言，可选覆盖率目标

## 6. Mock策略
所有用例需要mock以下目标：
- `pywrap_tfe.TFE_Py_FastPathExecute`：核心执行路径
- `_execute.record_gradient`：梯度记录
- 通过monkeypatch监控`_core._NotOkStatusException`异常传播