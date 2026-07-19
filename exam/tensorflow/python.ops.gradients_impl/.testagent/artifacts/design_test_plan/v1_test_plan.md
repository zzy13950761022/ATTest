# tensorflow.python.ops.gradients_impl 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock gradients_util._GradientsHelper 等内部依赖
- 随机性处理：固定随机种子，控制张量生成
- 图上下文：确保在 TensorFlow 图模式下执行

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03（首轮生成）
- DEFERRED_SET: CASE_04, CASE_05（后续迭代）
- 测试文件路径：tests/test_tensorflow_python_ops_gradients_impl.py（单文件）
- 断言分级策略：首轮使用 weak 断言，最终启用 strong 断言
- 预算策略：每个用例 S 大小，max_lines=80-85，max_params=6

## 3. 数据与边界
- 正常数据集：float32/float64 张量，形状 [2,3]/[3,2]/[4,4] 等
- 随机生成策略：固定种子生成正态分布数据
- 边界值：零形状张量、整数张量、极端数值（inf/nan）
- 空输入：空列表、None 值、零长度维度
- 负例与异常场景：
  - 不同图中的张量
  - 形状不匹配的 grad_ys
  - 无效的 unconnected_gradients 枚举值
  - 非张量类型输入

## 4. 覆盖映射
| TC_ID | 需求/约束覆盖 | 优先级 |
|-------|---------------|--------|
| TC-01 | 基本梯度计算：单张量ys对单张量xs | High |
| TC-02 | 列表输入：多ys对多xs的梯度聚合 | High |
| TC-03 | 偏导数计算：使用stop_gradients参数 | High |
| TC-04 | 未连接梯度处理：none和zero策略 | High |
| TC-05 | 自定义初始梯度：通过grad_ys参数 | High |

### 尚未覆盖的风险点
- 多实体情况：模块包含 gradients_v2, hessians 等函数
- 布尔参数组合：colocate_gradients_with_ops 和 gate_gradients 交互
- 不同 aggregation_method 的影响
- 控制流操作（while_loop, cond）的梯度计算
- 整数张量自动视为常数的行为验证

## 5. 迭代策略
- **首轮 (round1)**：生成 SMOKE_SET 用例，使用 weak 断言，最多 5 个用例块
- **后续迭代 (roundN)**：仅修复失败用例块，每次最多处理 3 个块，提升 deferred 用例
- **最终轮 (final)**：启用 strong 断言，可选覆盖率检查

## 6. Mock 目标
所有用例都需要 mock `tensorflow.python.ops.gradients_util._GradientsHelper` 以确保测试隔离，避免依赖内部实现细节。