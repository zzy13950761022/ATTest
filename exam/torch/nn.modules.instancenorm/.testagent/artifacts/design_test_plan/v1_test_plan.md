# torch.nn.modules.instancenorm 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：每个测试用例独立实例化，固定随机种子
- 随机性处理：使用固定随机种子确保可重复性
- 设备兼容性：优先CPU测试，GPU测试作为扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07, CASE_08
- **group列表**: G1（核心实例归一化类）, G2（惰性版本与高级功能）
- **active_group_order**: G1, G2
- **断言分级策略**: 首轮使用weak断言（形状、数据类型、有限值），后续启用strong断言（数值正确性）
- **预算策略**: 每个用例size=S，max_lines=80，max_params=6，支持参数化

## 3. 数据与边界
- **正常数据集**: 随机生成符合形状要求的浮点张量
- **边界值测试**: 
  - 单样本输入（批次大小=1）
  - 小通道数（num_features=1）
  - 极端形状（大/小空间维度）
  - 无批次输入（自动添加批次维度）
- **负例与异常场景**:
  - num_features非正整数
  - eps非正浮点数
  - momentum超出[0,1]范围
  - 输入通道数不匹配
  - 输入维度不符合要求

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖功能点 |
|-------|----------|------------|
| TC-01 | 基本前向传播 | InstanceNorm2d基本计算 |
| TC-02 | affine参数功能 | 缩放和偏移参数学习 |
| TC-03 | 惰性版本推断 | LazyInstanceNorm自动推断num_features |
| TC-04 | track_running_stats | 运行统计量跟踪与更新 |
| TC-05 | 无批次输入处理 | 自动批次维度添加 |

**尚未覆盖的风险点**:
- GPU设备兼容性（作为参数扩展）
- 混合精度训练支持
- 序列化/反序列化（state_dict）
- 梯度计算正确性验证
- 极端数值稳定性（NaN/Inf处理）