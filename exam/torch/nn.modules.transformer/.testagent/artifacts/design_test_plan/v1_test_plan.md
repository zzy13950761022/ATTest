# torch.nn.modules.transformer 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（控制随机种子、设备检测）
- 随机性处理：固定随机种子（控制 dropout 随机性）
- 设备隔离：CPU 优先，GPU 作为扩展参数

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_04, CASE_05, CASE_10（5个核心用例）
- **DEFERRED_SET**: CASE_03, CASE_06, CASE_07, CASE_08, CASE_09（5个延期用例）
- **group 列表**: G1（核心类）、G2（组件）、G3（高级功能）
- **active_group_order**: G1 → G2 → G3（按复杂度递增）
- **断言分级策略**: 首轮仅 weak 断言（形状、类型、有限性检查）
- **预算策略**: 
  - XS: 60行/4参数
  - S: 80行/6参数  
  - M: 90-100行/7-8参数
  - 所有用例均参数化，减少重复代码

## 3. 数据与边界
- **正常数据集**: 随机生成符合形状的张量，固定随机种子确保可重现
- **边界值**: seq_len=1, batch_size=1, d_model=最小整除值
- **极端形状**: 长序列（内存边界）、大批次（计算边界）
- **空输入**: 不支持（seq_len>0, batch_size>0 为硬约束）
- **负例场景**: 
  - d_model 不能被 nhead 整除 → ValueError
  - 无效激活函数字符串 → ValueError
  - 张量维度不匹配 → RuntimeError
  - 无效掩码类型/形状 → RuntimeError

## 4. 覆盖映射
| TC ID | 需求覆盖 | 约束覆盖 | 优先级 |
|-------|----------|----------|--------|
| TC-01 | 标准Transformer前向传播 | 形状、类型、设备一致性 | High |
| TC-02 | 仅编码器模式 | tgt=None 处理 | High |
| TC-04 | 参数验证异常 | d_model整除性验证 | High |
| TC-05 | 编码器基础功能 | 层堆栈、前向传播 | High |
| TC-10 | 掩码处理基础 | BoolTensor掩码应用 | High |
| TC-03 | batch_first维度 | 维度顺序一致性 | Medium |
| TC-06 | 解码器基础功能 | 内存注意力机制 | High |
| TC-07 | 编码器单层 | 残差连接、层归一化 | Medium |
| TC-08 | 解码器单层 | 自注意力+交叉注意力 | Medium |
| TC-09 | 组件组合验证 | 端到端等价性 | Medium |

## 5. 尚未覆盖的风险点
- 嵌套张量优化路径触发条件
- 自定义编码器/解码器接口边界
- 混合精度训练数值稳定性
- 设备间迁移（CPU↔GPU）一致性
- 极端数值（inf/nan）传播行为
- 训练/推理模式切换完整影响

## 6. 迭代策略
- **首轮**: 仅生成 SMOKE_SET（5个用例），使用 weak 断言
- **后续轮次**: 修复失败用例，逐步启用 DEFERRED_SET
- **最终轮**: 启用 strong 断言，可选覆盖率目标
- **参数扩展**: Medium/Low 优先级作为已有 High CASE 的参数维度扩展