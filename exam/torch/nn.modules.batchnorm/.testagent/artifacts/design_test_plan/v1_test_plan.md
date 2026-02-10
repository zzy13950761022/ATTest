# torch.nn.modules.batchnorm 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用pytest fixtures管理模块实例，固定随机种子确保可重复性
- 随机性处理：固定torch随机种子，控制RNG状态
- 设备隔离：优先使用CPU测试，避免GPU依赖

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (BatchNorm1d基础前向传播), CASE_02 (训练评估模式切换), CASE_05 (affine=False配置)
- **DEFERRED_SET**: CASE_03 (BatchNorm3d基础功能), CASE_04 (懒加载类延迟初始化), CASE_06-CASE_08 (待定义)
- **group列表**: 
  - G1: 基础前向传播与模式切换 (BatchNorm1d/2d/3d)
  - G2: 参数配置与边界验证 (affine/track_running_stats/momentum配置)
- **active_group_order**: ["G1", "G2"] - 按组顺序生成测试
- **断言分级策略**: 首轮使用weak断言（形状、类型、有限性、基本属性），后续启用strong断言（近似相等、统计正确性）
- **预算策略**: 
  - S大小: max_lines=70-75, max_params=6
  - M大小: max_lines=85-90, max_params=5
  - 参数化测试优先，减少重复代码

## 3. 数据与边界
- **正常数据集**: 随机正态分布张量，形状符合各维度类要求
- **随机生成策略**: 固定种子确保可重复性，批量大小N≥2
- **边界值测试**:
  - 批量大小N=1（小批量统计不稳定）
  - 极端eps值（1e-10, 1.0）
  - momentum=None（累积移动平均）
  - 输入值全为0或常数
  - 输入值极大/极小（数值稳定性）
- **形状边界**:
  - BatchNorm1d: (N,C) 和 (N,C,L) 两种形式
  - BatchNorm2d: 最小空间尺寸1x1
  - BatchNorm3d: 最小空间尺寸1x1x1
- **负例与异常场景**:
  - num_features ≤ 0触发ValueError
  - 输入维度不符合类要求
  - eps ≤ 0触发ValueError
  - momentum超出[0,1]范围
  - 输入类型非Tensor

## 4. 覆盖映射
| TC_ID | 需求/约束覆盖 | 优先级 |
|-------|--------------|--------|
| TC-01 | BatchNorm1d基础功能，输入形状验证 | High |
| TC-02 | 训练/评估模式切换，统计量更新 | High |
| TC-03 | BatchNorm3d功能，3D输入处理 | High |
| TC-04 | 懒加载类延迟初始化机制 | Medium |
| TC-05 | affine=False配置，无学习参数 | High |

**尚未覆盖的风险点**:
- SyncBatchNorm分布式环境测试
- 多GPU并行训练场景
- 与自动混合精度（AMP）的兼容性
- 梯度检查与反向传播验证
- 序列化/反序列化状态恢复
- 极端数值稳定性测试（大eps值）