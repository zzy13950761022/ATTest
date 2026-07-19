# tensorflow.python.compiler.xla.xla 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 fixtures 管理 TensorFlow 会话，mock XLA 编译失败场景
- 随机性处理：固定随机种子 tf.random.set_seed(42)，控制 RNG 状态
- 设备隔离：测试仅限 CPU 设备，避免 GPU/TPU 依赖

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基本编译功能）、CASE_02（None与空输入处理）、CASE_03（单值输出包装）
- **DEFERRED_SET**: CASE_04（嵌套输入结构转换）、CASE_05-CASE_08（待后续生成）
- **group 列表**: 
  - G1: 核心编译功能（compile, _compile_internal）
  - G2: 特殊场景与边界处理（compile）
- **active_group_order**: ["G1", "G2"]（先测核心功能，再测特殊场景）
- **断言分级策略**: 首轮仅使用 weak 断言（形状、类型、基本相等性），后续启用 strong 断言（数值相等性、梯度检查）
- **预算策略**: 
  - size: S（小型测试，70-80行）
  - max_lines: 60-80行
  - max_params: 4-6个参数
  - 所有用例均为参数化测试

## 3. 数据与边界
- **正常数据集**: 随机生成浮点张量（float32/float64），形状 [2,3] 到 [10,10]
- **边界值**: 
  - 空输入：inputs=None 与 inputs=[]
  - 单值输出：验证元组包装
  - 零维张量：标量处理
  - 嵌套结构：列表/字典嵌套张量
- **极端形状**: 
  - 大维度：[100, 100] 矩阵运算
  - 零尺寸：[0, 5] 空张量边界
- **数值边界**: 
  - inf/nan 值传播
  - 极大/极小浮点数
- **负例与异常场景**:
  - 不支持操作异常（Placeholder、Summary操作）
  - 非可调用 computation 参数
  - 无效输入类型
  - XLA 编译失败模拟

## 4. 覆盖映射
| TC_ID | 需求/约束覆盖 | 优先级 |
|-------|--------------|--------|
| TC-01 | 基本 compile() 调用，有效 computation 和 inputs | High |
| TC-02 | None 输入处理，inputs=None 和 inputs=[] 等价性 | High |
| TC-03 | 特殊输出验证：单值输出包装为元组 | High |
| TC-04 | 嵌套输入结构转换测试 | High |
| TC-05 | 不支持操作异常触发（deferred） | Medium |
| TC-06 | 仅操作输出返回 NoOp（deferred） | Medium |
| TC-07 | None 输出处理（deferred） | Medium |
| TC-08 | 随机数操作语义验证（deferred） | Low |

## 5. 尚未覆盖的风险点
- compile() 函数已弃用状态，可能在未来版本移除
- 缺少详细类型注解，类型错误处理不明确
- 随机数操作在 XLA 编译中的语义差异
- 文档示例不完整，使用模式不清晰
- 错误处理信息有限，异常类型不明确
- 全局状态修改（变量作用域）的副作用
- 多设备兼容性（GPU/TPU）未测试

## 6. 迭代策略
- **首轮（round1）**: 仅生成 SMOKE_SET（3个用例），使用 weak 断言
- **后续（roundN）**: 修复失败用例，每次最多处理3个block，提升deferred用例
- **最终（final）**: 启用 strong 断言，可选覆盖率检查

## 7. 测试文件组织
- 主文件：`tests/test_tensorflow_python_compiler_xla_xla.py`
- 分组文件：
  - G1: `tests/test_tensorflow_python_compiler_xla_xla_core.py`
  - G2: `tests/test_tensorflow_python_compiler_xla_xla_special.py`
- 所有用例通过 BLOCK_ID（CASE_XX）唯一标识，与 TC_ID 一一对应