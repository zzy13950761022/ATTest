# torch.nn.utils.clip_grad 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 设备支持：CPU（必需），CUDA（可选）
- 原地修改验证：梯度张量前后对比

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07, CASE_08
- **group 列表**:
  - G1: clip_grad_norm_ 核心功能 (CASE_01, CASE_02, CASE_05, CASE_06)
  - G2: clip_grad_value_ 与弃用函数 (CASE_03, CASE_04, CASE_07, CASE_08)
- **active_group_order**: G1, G2
- **断言分级策略**: 首轮使用 weak 断言，最终轮启用 strong 断言
- **预算策略**:
  - size: S (小型用例)
  - max_lines: 60-85 行
  - max_params: 5-8 个参数
  - is_parametrized: 多数用例支持参数化

## 3. 数据与边界
- **正常数据集**: 随机梯度张量，形状 [2,3] 到 [4,4]，dtype float32/float64
- **边界值**: 零范数梯度，极大/极小梯度值，空梯度列表
- **极端形状**: 高维张量 [3,3,3]，大张量（性能测试）
- **空输入**: 无梯度参数列表，返回 torch.tensor(0.)
- **负例与异常场景**:
  - max_norm <= 0 参数错误
  - clip_value <= 0 参数错误
  - 非张量参数类型错误
  - 不支持 norm_type 值
  - error_if_nonfinite=True 且梯度包含 NaN/inf

## 4. 覆盖映射
| TC ID | 功能覆盖 | 需求/约束 | 优先级 |
|-------|----------|-----------|--------|
| TC-01 | clip_grad_norm_ 基本功能 | 梯度范数裁剪，原地修改 | High |
| TC-02 | clip_grad_norm_ 多范数类型 | norm_type=1,2,inf 支持 | High |
| TC-03 | clip_grad_value_ 基本功能 | 梯度值裁剪到 [-clip, clip] | High |
| TC-04 | clip_grad_norm 弃用警告 | 弃用函数警告行为 | High |
| TC-05 | 非有限梯度处理 | error_if_nonfinite 两种状态 | High |

## 5. 尚未覆盖的风险点
- 分布式训练场景未覆盖
- 梯度稀疏性处理未明确
- 内存使用峰值未定义
- 线程安全性未说明
- error_if_nonfinite 默认值未来变化风险

## 6. 迭代策略
- **首轮 (round1)**: 仅生成 SMOKE_SET (4个用例)，使用 weak 断言
- **后续轮 (roundN)**: 修复失败用例，提升 DEFERRED_SET，每次最多3个用例
- **最终轮 (final)**: 启用 strong 断言，覆盖率可选

## 7. Mock 目标
- CASE_04: warnings.warn (弃用警告验证)
- 其他用例: 无 mock 需求，直接测试真实功能