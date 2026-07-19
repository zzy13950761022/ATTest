# torch.nn.modules.activation 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用fixtures管理测试数据，mock验证torch.nn.functional调用
- 随机性处理：固定随机种子（torch.manual_seed），控制RReLU训练模式随机性
- 设备兼容性：优先CPU测试，GPU测试作为可选扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (ReLU基础), CASE_02 (Sigmoid/Tanh), CASE_03 (Softmax), CASE_04 (Hardtanh)
- **DEFERRED_SET**: CASE_05-CASE_13（覆盖其余激活函数和复杂场景）
- **group列表**: 
  - G1: 基础激活函数族（ReLU, Sigmoid, Tanh, LeakyReLU）
  - G2: Softmax与归一化族（Softmax, LogSoftmax, Softmin, Softsign）
  - G3: 阈值与分段函数族（Hardtanh, ReLU6, CELU, SELU, GELU）
  - G4: 复杂与特殊函数族（MultiheadAttention, RReLU, PReLU, Threshold）
- **active_group_order**: G1 → G2 → G3 → G4（按复杂度递增）
- **断言分级策略**: 首轮使用weak断言（形状、类型、有限值、基础属性），后续启用strong断言（梯度、数值稳定性、边界条件）
- **预算策略**: 
  - size: S（小型测试）
  - max_lines: 60-70行
  - max_params: 5-6个参数
  - 首轮只生成SMOKE_SET的4个用例

## 3. 数据与边界
- **正常数据集**: 标准正态分布随机数，形状[2,3,4]等中等维度
- **随机生成策略**: torch.randn + 固定种子，确保可重现性
- **边界值**: 
  - 零值输入（测试ReLU(0)等边界）
  - 极端大/小数值（±1e6, ±1e-6）
  - 特殊值（inf, -inf, NaN）
  - 空张量（torch.tensor([])）
  - 标量输入（0维张量）
- **负例与异常场景**:
  - Hardtanh的max_val <= min_val
  - Softmax的无效dim参数
  - MultiheadAttention参数不整除
  - 非张量输入类型错误
  - 不支持的数据类型

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖约束 | 优先级 |
|-------|----------|----------|--------|
| TC-01 | 基础正向传播正确性 | ReLU形状不变、非负性 | High |
| TC-02 | 基础正向传播正确性 | Sigmoid/Tanh值域验证 | High |
| TC-03 | 基础正向传播正确性 | Softmax归一化属性 | High |
| TC-04 | 参数边界验证 | Hardtanh阈值约束 | High |
| TC-05 | inplace操作测试 | LeakyReLU内存行为 | Medium |

**尚未覆盖的风险点**:
- MultiheadAttention优化路径条件复杂
- Softmax数值稳定性（大数值溢出）
- RReLU随机性难以完全控制
- 设备间转移（CPU↔GPU）兼容性
- 序列化/反序列化行为