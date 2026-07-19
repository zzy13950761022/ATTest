# tensorflow.python.ops.gen_bitwise_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用fixtures管理TensorFlow会话和资源
- 随机性处理：固定随机种子，使用确定性测试数据
- 执行模式：同时测试eager和graph两种执行模式
- 设备支持：优先CPU，GPU作为可选扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (bitwise_and基本功能), CASE_02 (bitwise_or形状广播), CASE_03 (invert有符号整数取反)
- **DEFERRED_SET**: CASE_04 (left_shift边界情况), CASE_05 (population_count基本功能)
- **测试文件路径**: tests/test_tensorflow_python_ops_gen_bitwise_ops.py
- **断言分级策略**: 首轮使用weak断言（形状、类型、非空、基本操作），后续启用strong断言（精确值、边界行为、一致性）
- **预算策略**: 每个用例S大小，最大70-80行，最多8个参数，全部参数化

## 3. 数据与边界
- **正常数据集**: 使用小尺寸确定值张量（2x3, 3x1等），覆盖所有支持数据类型
- **边界值**: 空张量、标量、大形状、零值、全1值、负数（有符号类型）
- **极端形状**: 单元素张量、高维张量、形状不匹配但可广播
- **空输入**: 零元素张量（shape含0维度）
- **负例场景**: 类型不匹配、不支持的数据类型、形状无法广播、移位操作越界
- **异常场景**: 有符号整数取反的负数表示、移位操作的实现定义行为

## 4. 覆盖映射
| TC ID | 对应需求/约束 | 覆盖函数 | 关键验证点 |
|-------|--------------|----------|------------|
| TC-01 | 数据类型兼容性、基本功能 | bitwise_and | 所有支持数据类型、形状匹配 |
| TC-02 | 形状广播功能 | bitwise_or | 广播规则、不同形状组合 |
| TC-03 | 有符号整数处理 | invert | 负数表示、位取反正确性 |
| TC-04 | 移位操作边界 | left_shift | 移位越界、实现定义行为 |
| TC-05 | 特殊函数验证 | population_count | 返回类型uint8、人口计数算法 |

## 5. 尚未覆盖的风险点
- GPU设备结果一致性（依赖硬件可用性）
- 大尺寸张量内存使用和性能
- 梯度计算支持情况（如适用）
- 与numpy结果的完全一致性验证
- 模块级导入和函数发现机制
- 装饰器（@tf_export, @_dispatch）的影响

## 6. 迭代策略
- **首轮**: 仅生成SMOKE_SET用例，使用weak断言，验证核心功能
- **后续轮次**: 修复失败用例，逐步启用DEFERRED_SET，添加参数扩展
- **最终轮**: 启用strong断言，添加覆盖率报告，验证边界情况

## 7. Mock策略
- 当前计划无需mock，直接使用TensorFlow运行时
- 如需隔离设备上下文，可mock `tensorflow.python.framework.ops.device`
- 如需控制执行模式，可mock `tensorflow.python.eager.context.executing_eagerly`
- 操作定义库 `tensorflow.python.ops.gen_bitwise_ops._op_def_library` 作为备用mock目标