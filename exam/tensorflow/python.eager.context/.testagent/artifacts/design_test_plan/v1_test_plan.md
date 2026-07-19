# tensorflow.python.eager.context 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 设备隔离：CPU-only 测试环境
- 状态管理：每个测试用例独立上下文

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_05, CASE_06
- **DEFERRED_SET**: CASE_04, CASE_07, CASE_08, CASE_09
- **group 列表**: G1（核心上下文函数族）, G2（Context类与设备策略）
- **active_group_order**: G1 → G2
- **断言分级策略**: 首轮使用 weak 断言，最终轮启用 strong 断言
- **预算策略**: 
  - size: S/M（小型/中型用例）
  - max_lines: 60-85 行
  - max_params: 3-6 个参数
  - 参数化用例优先

## 3. 数据与边界
- **正常数据集**: 标准设备策略和执行模式组合
- **随机生成策略**: 固定种子生成测试配置
- **边界值**: None 参数、空配置、无效枚举值
- **极端形状**: 多线程并发访问、重复初始化
- **空输入**: config=None, server_def=None
- **负例与异常场景**:
  - 无效 device_policy 值
  - 无效 execution_mode 值
  - 损坏的 config 对象
  - 多线程状态冲突
  - 环境变量干扰

## 4. 覆盖映射
| TC_ID | 需求/约束 | 优先级 | 状态 |
|-------|-----------|--------|------|
| TC-01 | executing_eagerly() 基础功能 | High | SMOKE |
| TC-02 | context_safe() 上下文获取 | High | SMOKE |
| TC-03 | ensure_initialized() 幂等性 | High | SMOKE |
| TC-04 | executing_eagerly() 在 tf.function 内部 | Medium | DEFERRED |
| TC-05 | Context 基础初始化 | High | SMOKE |
| TC-06 | Context 无效参数验证 | High | SMOKE |

## 5. 尚未覆盖的风险点
- **远程执行**: server_def 参数的具体使用方式
- **设备策略细节**: device_policy 默认行为可能随版本变化
- **执行模式默认值**: execution_mode 自动选择逻辑未完全文档化
- **多设备环境**: GPU/TPU 设备放置的并发行为
- **内存管理**: 资源释放时机和内存泄漏风险
- **环境变量**: TF_RUN_EAGER_OP_AS_FUNCTION 的具体影响

## 6. 迭代策略
- **首轮 (round1)**: 仅生成 SMOKE_SET 用例，使用 weak 断言
- **中间轮 (roundN)**: 修复失败用例，提升 DEFERRED 用例
- **最终轮 (final)**: 启用 strong 断言，可选覆盖率目标

## 7. 测试文件结构
- 主文件: `tests/test_tensorflow_python_eager_context.py`
- 分组文件: 
  - G1: `tests/test_tensorflow_python_eager_context_g1.py`
  - G2: `tests/test_tensorflow_python_eager_context_g2.py`
- 模式匹配: `tests/test_tensorflow_python_eager_context_*.py`