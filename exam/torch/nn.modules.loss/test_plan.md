# torch.nn.modules.loss 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：每个测试用例独立实例化损失对象，无状态共享
- 随机性处理：固定随机种子确保可重复性，使用 torch.manual_seed
- 设备策略：首轮仅测试CPU，后续扩展GPU兼容性

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (L1Loss基础), CASE_02 (MSELoss三种reduction), CASE_03 (CrossEntropyLoss形状), CASE_06 (BCELoss概率), CASE_07 (NLLLoss负对数似然)
- **DEFERRED_SET**: CASE_04 (弃用参数), CASE_05 (极端值), CASE_08 (KLDivLoss特殊), CASE_09 (加权损失), CASE_10 (设备兼容)
- **group列表**: 
  - G1: 核心损失函数族 (L1Loss, MSELoss, CrossEntropyLoss)
  - G2: 加权与特殊损失函数族 (BCELoss, NLLLoss, KLDivLoss)
- **active_group_order**: ["G1", "G2"] - 按优先级顺序执行
- **断言分级策略**: 首轮仅使用weak断言（形状、类型、有限性检查），后续启用strong断言（数值精度、边界情况）
- **预算策略**: 
  - S大小: max_lines≤80, max_params≤6
  - M大小: max_lines≤85, max_params≤6
  - 所有用例均支持参数化

## 3. 数据与边界
- **正常数据集**: 随机生成符合各损失函数要求的张量（如BCELoss需要0-1范围）
- **随机生成策略**: 使用torch.rand/randn，固定种子确保可重复性
- **边界值**: 
  - 空张量输入（零元素）
  - 单元素张量（最小形状）
  - 极端数值（inf, nan, 极大/极小值）
  - 零长度维度
- **负例与异常场景**:
  - 无效reduction值触发ValueError
  - 形状不匹配触发RuntimeError
  - 类型错误触发TypeError
  - 弃用参数触发警告
  - 概率输入超出[0,1]范围

## 4. 覆盖映射
| TC_ID | 需求/约束覆盖 | 优先级 |
|-------|--------------|--------|
| TC-01 | L1Loss基础功能，任意形状输入 | High |
| TC-02 | reduction三种模式正确性 | High |
| TC-03 | CrossEntropyLoss形状兼容性 | High |
| TC-04 | 弃用参数向后兼容性 | Medium |
| TC-05 | 极端形状与数值边界处理 | Medium |
| TC-06 | BCELoss概率输入验证 | High |
| TC-07 | NLLLoss负对数似然计算 | High |
| TC-08 | KLDivLoss特殊reduction处理 | Medium |
| TC-09 | 加权损失类权重参数 | Medium |
| TC-10 | 设备兼容性验证 | Low |

## 5. 尚未覆盖的风险点
- 复数输入支持范围不明确（仅L1Loss提及）
- 类型注解不完整导致的类型错误风险
- 形状约束描述模糊可能遗漏边缘情况
- 性能约束缺失，无法验证计算效率
- 所有20+损失函数的完整覆盖（首轮仅覆盖6个核心类）

## 6. 迭代策略
- **首轮**: 仅生成SMOKE_SET中的5个用例，使用weak断言
- **后续轮次**: 修复失败用例，从DEFERRED_SET提升优先级
- **最终轮**: 启用strong断言，可选覆盖率检查

## 7. 模块拆分策略
- **G1组**: 覆盖最常用的3个损失类，确保核心功能正确
- **G2组**: 覆盖需要特殊处理的损失类（概率输入、加权、特殊reduction）
- 每组有自己的SMOKE_SET和DEFERRED_SET，便于增量开发