# tensorflow.python.ops.custom_gradient 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 执行模式：图模式(graph)与急切模式(eager)双覆盖
- 梯度验证：使用 TensorFlow GradientTape 作为基准

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03
- DEFERRED_SET: CASE_04, CASE_05
- 测试文件路径：tests/test_tensorflow_python_ops_custom_gradient.py
- 断言分级策略：首轮使用 weak 断言，最终轮启用 strong 断言
- 预算策略：S/M 规模，max_lines 80-95，max_params 6-7

## 3. 数据与边界
- 正常数据集：随机生成浮点张量，形状 [2,2] 到 [5,5]
- 边界值：空张量、极端数值(log1pexp)、零梯度场景
- 数据类型：float32/float64 双精度覆盖
- 变量类型：ResourceVariable 正确性验证
- 负例场景：非元组返回、错误 grad_fn 签名、非 ResourceVariable

## 4. 覆盖映射
| TC ID | 需求覆盖 | 关键约束 |
|-------|----------|----------|
| TC-01 | 基本装饰器功能 | 输出值正确，梯度存在 |
| TC-02 | 模式一致性 | 图/急切模式行为一致 |
| TC-03 | 变量梯度传播 | ResourceVariable 支持 |
| TC-04 | 嵌套梯度场景 | 组合功能正确性 |
| TC-05 | 数值稳定性 | 边界值处理，无 NaN/Inf |

## 5. 尚未覆盖的风险点
- 非 ResourceVariable 变量处理细节
- grad_fn 返回梯度数量运行时验证
- 嵌套自定义梯度二阶行为复杂性
- 图模式 kwargs 参数限制边界
- 装饰器工厂模式(f=None)完整场景