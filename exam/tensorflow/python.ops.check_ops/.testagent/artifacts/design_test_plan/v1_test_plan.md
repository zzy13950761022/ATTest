# tensorflow.python.ops.check_ops 测试计划

## 1. 测试策略
- **单元测试框架**：pytest
- **隔离策略**：mock/monkeypatch/fixtures（需要 mock 执行模式检测、静态检查、比较操作等）
- **随机性处理**：固定随机种子，使用确定性张量生成
- **执行模式**：测试 eager/graph/静态检查三种模式
- **设备隔离**：仅测试 CPU 模式，避免 GPU 依赖

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03（核心路径：eager模式、graph模式、静态失败）
- **DEFERRED_SET**: CASE_04, CASE_05（空张量、广播形状）
- **单文件路径**: tests/test_tensorflow_python_ops_check_ops.py
- **断言分级策略**: 首轮使用 weak 断言（基本行为验证），后续启用 strong 断言（详细验证）
- **预算策略**: 每个用例 size=S，max_lines=80，max_params=6，支持参数化
- **Mock 策略**: 所有用例都需要 mock 核心依赖（constant_value、比较操作、Assert 操作等）

## 3. 数据与边界
- **正常数据集**: 使用固定种子生成数值张量，覆盖主要数据类型（float32, int32, float64, int64）
- **边界值**: 空张量（[0,0]）、零长度维度、广播兼容形状（[1,3,1] vs [3,1,5]）
- **负例与异常场景**:
  1. 静态检查失败（立即引发异常）
  2. 类型不匹配（TypeError）
  3. 形状不兼容（ValueError）
  4. 极端数值（inf, nan）
  5. 量化数据类型边界
  6. 复杂数据类型支持

## 4. 覆盖映射
| TC_ID | 需求约束 | 覆盖点 |
|-------|----------|--------|
| TC-01 | 高优先级1 | assert_equal 在 eager 模式下返回 None |
| TC-02 | 高优先级2 | assert_equal 在 graph 模式下创建 Assert 操作 |
| TC-03 | 高优先级3 | assert_less 静态检查失败引发异常 |
| TC-04 | 高优先级4 | assert_positive 空张量自动通过 |
| TC-05 | 高优先级5 | 广播形状正确处理 |

### 尚未覆盖的风险点
1. 稀疏张量支持（文档未明确说明）
2. 分布式环境行为差异
3. 自定义设备（TPU）支持
4. v1/v2 版本兼容性细节
5. 嵌套控制依赖复杂场景
6. 超大张量性能影响

## 5. 迭代说明
- **首轮（round1）**: 仅生成 SMOKE_SET 中的 3 个核心用例，使用 weak 断言
- **后续轮次（roundN）**: 修复失败用例，从 deferred_set 提升用例，每次最多 3 个新用例
- **最终轮次（final）**: 启用 strong 断言，可选覆盖扩展

## 6. 技术要点
- 所有用例都需要 mock 执行模式检测（executing_eagerly_outside_functions）
- 静态检查依赖 constant_value mock
- 动态断言创建依赖 control_flow_ops.Assert mock
- 比较操作（equal, less, greater 等）需要 mock 以验证调用
- 使用参数化测试覆盖不同函数和数据类型组合