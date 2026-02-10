# tensorflow.python.ops.script_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（针对graph模式、函数注册、设备复制）
- 随机性处理：固定随机种子，控制NumPy/TensorFlow RNG
- 执行模式：eager和graph模式双覆盖
- 设备隔离：CPU环境为主，避免GPU依赖

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (eager_py_func基础调用), CASE_02 (py_func_common NumPy转换), CASE_03 (numpy_function别名验证)
- **DEFERRED_SET**: CASE_04 (复合张量支持), CASE_05 (错误处理与类型检查)
- **测试文件路径**: tests/test_tensorflow_python_ops_script_ops.py（单文件）
- **断言分级策略**: 首轮使用weak断言（形状、类型、基础操作验证），后续启用strong断言（精度、梯度、设备一致性）
- **预算策略**: 
  - Size S: 65-80行，4-6参数
  - Size M: 90行，7参数（仅CASE_04）
  - 参数化：除CASE_03外均为参数化测试

## 3. 数据与边界
- **正常数据集**: 随机生成浮点/整数张量，形状[2-5, 2-5]，使用固定种子
- **边界值**: 
  - 空输入列表（func无参数）
  - 零维标量张量
  - 极端形状[1, 1000]或[1000, 1]
  - 空字符串张量
- **复合张量边界**:
  - RaggedTensor不规则形状
  - SparseTensor稀疏表示
  - 嵌套结构输入
- **负例与异常场景**:
  - 无效func（非callable）
  - 类型不匹配（Tout错误）
  - 输入非列表
  - stateful参数误用
  - 设备上下文错误

## 4. 覆盖映射
| TC ID | 需求覆盖 | 约束覆盖 | Mock目标 |
|-------|----------|----------|----------|
| TC-01 | eager_py_func基础调用 | Tensor输入输出，eager模式 | 无 |
| TC-02 | py_func_common NumPy转换 | graph模式，NumPy数组转换 | FuncGraph, get_default_graph |
| TC-03 | numpy_function别名验证 | API一致性，字符串处理 | 无 |
| TC-04 | 复合张量支持 | CompositeTensor包装，设备复制 | _wrap_for_composites, _maybe_copy_to_context_device |
| TC-05 | 错误处理 | 类型检查，异常抛出 | 无 |

**尚未覆盖的风险点**:
- GIL获取对并发性能影响（性能测试范围外）
- 函数体不序列化到GraphDef（序列化测试范围外）
- 不支持跨进程分布式（分布式测试范围外）
- 异步执行限制（异步测试范围外）

## 5. 迭代策略
- **首轮**: 仅生成SMOKE_SET（3个用例），使用weak断言
- **后续轮次**: 修复失败用例，从DEFERRED_SET提升用例（每次最多3个）
- **最终轮**: 启用strong断言，可选覆盖率检查

## 6. 验证参考
- NumPy等效操作作为oracle（TC-01, TC-02）
- 直接NumPy调用对比（TC-02）
- py_func_common行为一致性（TC-03）
- 手动复合张量处理（TC-04）
- Python标准异常（TC-05）