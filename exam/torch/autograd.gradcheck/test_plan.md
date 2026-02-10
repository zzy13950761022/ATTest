# torch.autograd.gradcheck 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 fixtures 管理测试函数和输入数据，mock 异常处理路径
- 随机性处理：固定随机种子确保可重复性，控制有限差分扰动
- 设备管理：支持 CPU 和 CUDA 设备测试

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基本实数函数梯度验证）、CASE_02（复数函数Wirtinger导数检查）、CASE_03（稀疏张量梯度检查）
- **DEFERRED_SET**: CASE_04（前向模式自动微分验证）、CASE_05（异常处理）、CASE_06（快速模式验证）
- **group 列表**: G1（基本功能验证）、G2（高级功能与异常处理）
- **active_group_order**: ["G1", "G2"]
- **断言分级策略**: 首轮使用 weak 断言（returns_bool, no_exception, basic_gradient_check），后续启用 strong 断言（gradient_accuracy, tolerance_validation）
- **预算策略**: 每个 CASE size=S，max_lines=60-75，max_params=8-9，is_parametrized=true

## 3. 数据与边界
- **正常数据集**: 简单多项式函数（x², x³），复数函数（z·conj(z)），随机生成小尺寸张量（2x2, 3x3, 4x4）
- **边界值**: 单元素张量，零维度张量，极端数值（inf, nan），高维张量（>4维）
- **极端形状**: 大尺寸稀疏矩阵（6x6），非连续内存张量，重叠内存张量
- **负例与异常场景**:
  - 输入张量未设置 requires_grad=True
  - 函数返回非张量类型
  - 复数函数在 fast_mode=True 时
  - 容差参数非法值（负值、零值）
  - 不支持的设备或数据类型

## 4. 覆盖映射
| TC_ID | 需求/约束 | 优先级 | 断言级别 |
|-------|-----------|--------|----------|
| TC-01 | 基本实数函数梯度验证 | High | weak |
| TC-02 | 复数函数Wirtinger导数检查 | High | weak |
| TC-03 | 稀疏张量梯度检查 | High | weak |
| TC-04 | 前向模式自动微分验证 | Medium | weak |
| TC-05 | 异常处理：raise_exception行为 | Medium | weak |
| TC-06 | 快速模式验证 | Medium | weak |

**尚未覆盖的风险点**:
- 重叠内存张量行为未详细说明
- 不同精度张量的具体容差要求不明确
- 非确定性容差（nondet_tol）的具体应用场景
- 批处理梯度检查（check_batched_grad=True）
- 梯度数据类型检查（check_grad_dtypes=True）

## 5. 迭代策略
- **首轮（round1）**: 仅生成 SMOKE_SET（3个核心用例），使用 weak 断言
- **后续轮次（roundN）**: 修复失败用例，从 DEFERRED_SET 提升用例，每次最多3个用例
- **最终轮次（final）**: 启用 strong 断言，可选覆盖率检查

## 6. 测试文件组织
- 默认文件: `tests/test_torch_autograd_gradcheck.py`
- 分组文件: 
  - G1: `tests/test_torch_autograd_gradcheck_basic.py`（基本功能）
  - G2: `tests/test_torch_autograd_gradcheck_advanced.py`（高级功能）
- 所有测试文件模式: `tests/test_torch_autograd_gradcheck_*.py`