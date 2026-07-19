# tensorflow.python.framework.tensor_util 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock fast_tensor_util 降级路径，使用 fixtures 管理测试数据
- 随机性处理：固定 numpy 随机种子，控制测试数据生成
- 测试分组：按功能拆分为 2 个 group，分别测试核心功能和边界异常

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01（基本数据类型转换）、CASE_02（形状验证与广播互斥）、CASE_03（TensorProto输入直接返回）
- DEFERRED_SET: CASE_04（特殊数据类型支持）等 5 个用例
- group 列表：G1（核心转换功能）、G2（边界与异常处理）
- active_group_order: ["G1", "G2"] - 优先测试核心功能
- 断言分级策略：首轮仅使用 weak 断言（类型匹配、形状匹配、值保留）
- 预算策略：size=S/M，max_lines=60-80，max_params=3-6，优先参数化测试

## 3. 数据与边界
- 正常数据集：Python 列表、numpy 数组、标量值
- 随机生成策略：小尺寸固定种子数组，覆盖主要数据类型
- 边界值：空列表、0 长度数组、单元素数组
- 极端形状：2D/3D 小矩阵，避免内存问题
- 空输入：None 值处理（需验证异常）
- 负例场景：形状不匹配、类型不兼容、广播冲突
- 异常场景：NaN/Inf 浮点数、超出范围整数值

## 4. 覆盖映射
- TC-01 → 需求 4.1：基本数据类型转换（int32, float32, bool）
- TC-02 → 需求 4.2：形状验证与广播功能互斥性
- TC-03 → 需求 4.4：TensorProto 输入直接返回
- TC-04 → 需求 4.3：特殊数据类型支持（float16, bfloat16）

### 尚未覆盖的风险点
- fast_tensor_util 不可用时的降级处理
- 大尺寸张量内存限制测试
- 并发调用安全性验证
- 不同 numpy 版本兼容性
- 遗留 TensorFlow 1.x 工作流完整性