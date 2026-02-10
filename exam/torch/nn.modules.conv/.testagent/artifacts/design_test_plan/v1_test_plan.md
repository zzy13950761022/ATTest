# torch.nn.modules.conv 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 pytest fixtures 管理模块实例，mock torch.nn.functional.conv* 用于验证调用
- 随机性处理：固定随机种子（torch.manual_seed），控制权重初始化确定性

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04（4个核心用例）
- DEFERRED_SET: CASE_05, CASE_06, CASE_07, CASE_08, CASE_09（5个延期用例）
- group 列表：G1（Conv2d核心功能测试）、G2（卷积变体与高级功能）
- active_group_order: ["G1", "G2"]（先测G1再测G2）
- 断言分级策略：首轮使用weak断言（模块实例化、前向传播、输出形状、数据类型）
- 预算策略：size=S/M（小/中），max_lines=60-85行，max_params=6-8个

## 3. 数据与边界
- 正常数据集：随机生成符合形状约束的Tensor，使用torch.randn
- 边界值：in_channels=1/out_channels=1，kernel_size=1，极端形状（接近内存限制）
- 空输入：不支持空Tensor，维度为0时抛出异常
- 负例场景：groups不整除、无效padding_mode、padding='same'且stride≠1
- 极端数值：极大/极小浮点数，NaN/Inf检查

## 4. 覆盖映射
| TC_ID | 需求/约束覆盖 | 风险点 |
|-------|--------------|--------|
| TC-01 | 基本Conv2d实例化与前向传播 | 输出形状计算正确性 |
| TC-02 | 参数验证与异常处理 | 错误消息准确性 |
| TC-03 | Conv1d/Conv3d基本功能 | 维度一致性 |
| TC-04 | padding='same'与特殊模式 | same padding输出形状精确性 |
| TC-05 | 权重初始化验证 | 浮点精度问题 |

## 5. 尚未覆盖的关键风险点
- 复数数据类型支持的具体限制
- CUDA非确定性算法行为
- 设备间迁移（CPU↔CUDA）的正确性
- 序列化/反序列化后的状态一致性
- 梯度计算正确性（需要autograd测试）

## 6. 首轮执行策略
1. 仅生成SMOKE_SET中的4个用例
2. 使用weak断言级别
3. 优先测试G1组（Conv2d核心功能）
4. 每个用例保持简洁，不超过85行代码
5. 参数化测试减少重复代码

## 7. 后续迭代策略
- roundN：修复失败用例，每次最多处理3个block
- 从deferred_set提升用例到smoke_set
- final轮：启用strong断言，可选覆盖率检查