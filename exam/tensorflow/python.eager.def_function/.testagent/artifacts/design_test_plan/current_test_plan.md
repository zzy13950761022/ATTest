# tensorflow.python.eager.def_function 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures 用于变量初始化和重跟踪计数
- 随机性处理：固定随机种子，控制张量生成范围
- 参考实现：eager 模式执行作为 oracle

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07, CASE_08
- **group 列表**: G1（核心装饰器与基本功能）, G2（高级特性与边界情况）
- **active_group_order**: G1, G2
- **断言分级策略**: 首轮使用 weak 断言，最终轮启用 strong 断言
- **预算策略**: 
  - size: S/M（小型/中型）
  - max_lines: 60-85 行
  - max_params: 5-6 个参数
  - is_parametrized: 首轮为 false，后续作为参数扩展

## 3. 数据与边界
- **正常数据集**: 简单数学运算、状态计数器、条件分支函数
- **随机生成策略**: 固定种子生成小规模张量（2x2, 标量）
- **边界值**: 
  - input_signature=None（无约束模式）
  - input_signature=空序列
  - autograph=False（禁用 AutoGraph）
  - jit_compile=True（启用 XLA）
  - func=None（装饰器工厂模式）
- **极端形状**: 0维标量、2x2小矩阵
- **空输入**: 无参数函数、空闭包
- **负例与异常场景**:
  - 非可调用对象作为 func
  - 无效 TensorSpec 格式
  - 类型不匹配输入
  - 已弃用参数警告

## 4. 覆盖映射
| TC_ID | 需求/约束 | 风险点 |
|-------|-----------|--------|
| TC-01 | 基本装饰器用法、返回 GenericFunction | 装饰器语法兼容性 |
| TC-02 | input_signature 限制重跟踪 | 重跟踪条件边界 |
| TC-03 | 变量创建与状态保持 | 变量初始化限制 |
| TC-04 | 控制流语句支持 | AutoGraph 转换边界 |
| TC-05 | 装饰器工厂模式 | 参数传递机制 |

**尚未覆盖的关键风险点**:
1. experimental_compile 已弃用但需向后兼容
2. 变量初始化限制的具体边界条件
3. 重跟踪性能影响量化验证
4. XLA 编译兼容性问题
5. 类型注解支持的完整范围

## 5. 迭代策略
- **首轮 (round1)**: 仅生成 SMOKE_SET（4个用例），使用 weak 断言
- **后续轮 (roundN)**: 修复失败用例，提升 deferred 用例，每次最多3个块
- **最终轮 (final)**: 启用 strong 断言，可选覆盖率检查

## 6. 模块拆分
- **G1**: 核心装饰器与基本功能（CASE_01, CASE_02, CASE_05）
- **G2**: 高级特性与边界情况（CASE_03, CASE_04）

每个 group 有自己的 SMOKE_SET 和 DEFERRED_SET，确保模块化测试和渐进式覆盖。