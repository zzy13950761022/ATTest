# torch.nn.modules.dropout 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用固定随机种子控制随机性，mock torch.nn.functional.dropout* 函数用于验证调用
- 随机性处理：固定随机种子确保测试可重复，使用统计方法验证随机模式
- 设备支持：首轮仅测试CPU，后续扩展CUDA

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04（共4个核心用例）
- **DEFERRED_SET**: CASE_05至CASE_10（6个延期用例）
- **group列表**: G1（核心Dropout类）、G2（高级Dropout类）、G3（边界与异常）
- **active_group_order**: G1 → G2 → G3（按优先级顺序）
- **断言分级策略**: 首轮仅使用weak断言（形状、类型、基本属性），后续启用strong断言（精确缩放、统计验证）
- **预算策略**: 每个用例size=S/M，max_lines=65-85行，max_params=4-6个参数

## 3. 数据与边界
- **正常数据集**: 随机生成符合各Dropout类形状约束的张量，使用固定种子确保可重复
- **边界值测试**:
  - p=0（无dropout，恒等映射）
  - p=1（全部置零，训练模式）
  - 空张量和零维度输入
  - 极端形状（单元素、超大维度）
- **负例与异常场景**:
  - p<0或p>1触发ValueError
  - 不支持的输入形状触发RuntimeError
  - Dropout2d对3D输入的警告
  - inplace操作的副作用验证

## 4. 覆盖映射
- **TC-01**: 覆盖训练/评估模式切换需求，验证p参数基本功能
- **TC-02**: 覆盖Dropout1d形状约束，验证通道维度保持
- **TC-03**: 覆盖AlphaDropout统计特性，验证零均值和单位方差
- **TC-04**: 覆盖参数边界值，验证p=0和p=1的特殊行为

- **尚未覆盖的关键风险点**:
  1. Dropout2d对3D输入的特殊行为（历史兼容性）
  2. 随机性测试的统计可靠性（需要大量样本）
  3. CUDA设备支持与CPU一致性
  4. 内存使用和原地操作性能影响
  5. AlphaDropout与SELU激活函数的兼容性验证

## 5. 迭代计划
- **首轮**: 仅生成SMOKE_SET中的4个用例，使用weak断言
- **后续轮次**: 修复失败用例，从DEFERRED_SET中提升优先级，添加参数扩展
- **最终轮**: 启用strong断言，可选覆盖率检查，完善所有边界场景