# tensorflow.python.ops.functional_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 测试模式：支持 eager 和 graph 模式
- 设备兼容：CPU 优先，GPU 可选

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01 (foldl基本折叠), CASE_02 (scan累积序列), CASE_03 (If条件分支)
- DEFERRED_SET: CASE_04 (While循环), CASE_05 (嵌套结构)
- 测试文件路径：tests/test_tensorflow_python_ops_functional_ops.py
- 断言分级策略：首轮使用 weak 断言，最终轮启用 strong 断言
- 预算策略：size=S, max_lines=80, max_params=6
- 迭代策略：首轮5个核心用例，后续修复失败块，最终启用强断言

## 3. 数据与边界
- 正常数据集：小规模张量（形状[6]、[5]、[3,3]等），float32/int32 dtype
- 随机生成策略：固定种子生成可重复测试数据
- 边界值：空张量（有initializer）、零维张量、极大序列长度
- 极端形状：深度嵌套结构、不同dtype混合
- 空输入：elems为空且无initializer（应抛异常）
- 负例场景：fn不可调用、elems第一维不匹配、不支持dtype

## 4. 覆盖映射
- TC-01 (CASE_01): foldl基本折叠操作正确性（需求1）
- TC-02 (CASE_02): scan累积序列生成验证（需求2）
- TC-03 (CASE_03): If条件分支正确执行（需求3）
- TC-04 (CASE_04): While循环控制流测试（需求4）
- TC-05 (CASE_05): 嵌套结构和多参数支持（需求5）

## 5. 尚未覆盖的风险点
- parallel_iterations 参数并行效果
- swap_memory 内存交换具体触发条件
- 图模式与 eager 模式行为差异
- 捕获输入（captured_inputs）处理逻辑
- 零维张量边界情况处理
- 不同 dtype 混合输入支持程度
- 嵌套结构深度限制未文档化

## 6. Mock 目标
- tensorflow.python.ops.gen_functional_ops（底层C++操作）
- tensorflow.python.ops.control_flow_ops.while_loop（循环实现）
- 仅用于需要控制底层行为的测试用例（CASE_03, CASE_04）

## 7. 验证参考
- foldl/scan: numpy.cumsum 作为参考实现
- If/While: 手动验证逻辑正确性
- 梯度计算：TensorFlow 自动微分验证
- 形状推断：TensorFlow 形状系统验证