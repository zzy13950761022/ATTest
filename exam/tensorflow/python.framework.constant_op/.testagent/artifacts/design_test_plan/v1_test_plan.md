# tensorflow.python.framework.constant_op 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 fixtures 管理 TensorFlow 会话和 eager 模式切换
- 随机性处理：无随机性要求，使用固定测试数据
- 设备隔离：测试在当前设备（CPU）上运行，不涉及 GPU 特定逻辑

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04（共4个核心用例）
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07, CASE_08（共4个延期用例）
- **group 列表**:
  - G1: 核心函数 constant() 测试（包含 CASE_01, CASE_02, CASE_05, CASE_06）
  - G2: 兼容函数 constant_v1() 测试（包含 CASE_03, CASE_04, CASE_07, CASE_08）
- **active_group_order**: G1, G2（按优先级顺序执行）
- **断言分级策略**: 首轮使用 weak 断言（shape, dtype, value_match, tensor_type），后续启用 strong 断言
- **预算策略**: 
  - size: S（小型测试，60-80行）
  - max_lines: 60-80行
  - max_params: 4-5个参数
  - is_parametrized: true（使用参数化测试）
  - requires_mock: false（首轮不需要mock）

## 3. 数据与边界
- **正常数据集**: Python 标量、列表、嵌套列表、numpy 数组
- **随机生成策略**: 不使用随机数据，使用固定测试用例
- **边界值**:
  - 空列表 [] → 形状 (0,) 的张量
  - 标量 0 → 形状 () 的张量
  - 单元素列表 [42] → 形状 (1,) 的张量
- **极端形状**: 大维度数组（延期测试）
- **空输入**: 空列表和零值标量
- **负例与异常场景**:
  - 无效 shape 参数（负值、非整数）
  - 不兼容的 shape 和 value
  - 无效 dtype 类型
  - verify_shape=True 时形状不匹配

## 4. 覆盖映射
- **TC-01 (CASE_01)**: 覆盖基本标量和列表创建功能
- **TC-02 (CASE_02)**: 覆盖 dtype 显式指定和自动推断
- **TC-03 (CASE_03)**: 覆盖 constant_v1() 基本功能
- **TC-04 (CASE_04)**: 覆盖 verify_shape 参数验证
- **TC-05 (CASE_05)**: 覆盖 shape 重塑和广播功能

- **尚未覆盖的风险点**:
  - 符号张量处理（明确排除在范围外）
  - 广播规则的详细约束
  - 缓存机制的具体实现
  - 设备放置的默认策略
  - 大尺寸张量创建性能
  - 特殊数据类型（复数、字符串）