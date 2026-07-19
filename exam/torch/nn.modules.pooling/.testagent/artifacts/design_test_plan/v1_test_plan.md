# torch.nn.modules.pooling 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 fixtures 管理测试数据，mock FractionalMaxPool 的随机数生成器
- 随机性处理：固定随机种子控制 FractionalMaxPool 的随机池化区域选择
- 设备隔离：分别测试 CPU 和 CUDA 设备（首轮仅 CPU）

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (MaxPool1d基本功能), CASE_02 (AvgPool2d基本功能), CASE_05 (AdaptiveAvgPool2d基本功能), CASE_08 (FractionalMaxPool2d基本功能)
- **DEFERRED_SET**: CASE_03, CASE_04, CASE_06, CASE_07, CASE_09, CASE_10
- **group 列表**: 
  - G1: 基础池化功能 (MaxPool1d/2d/3d, AvgPool1d/2d/3d)
  - G2: 高级池化功能 (AdaptiveMax/AvgPool1d/2d/3d)
  - G3: 特殊池化功能 (FractionalMaxPool2d/3d, LPPool1d/2d, MaxUnpool1d/2d/3d)
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮仅使用 weak 断言（实例创建、前向传播、输出形状、数据类型）
- **预算策略**: 
  - size: S (小型测试) / M (中型测试)
  - max_lines: 50-80 行
  - max_params: 5-9 个参数
  - 首轮 4 个 SMOKE 用例，总代码量约 250 行

## 3. 数据与边界
- **正常数据集**: 随机生成浮点张量，形状符合池化维度要求
- **随机生成策略**: 使用固定种子确保可重复性，范围 [-10, 10]
- **边界值测试**:
  - kernel_size=1 (最小窗口)
  - 输入尺寸=1x1 (最小输入)
  - padding=0 和最大有效 padding
  - ceil_mode=True 时的边界情况
  - dilation=1 (最小膨胀)
- **极端形状**:
  - 大尺寸输入（内存边界）
  - 非方形输入（长宽不等）
  - 单通道/多通道输入
- **空输入**: 零尺寸维度（后续轮次）
- **负例与异常场景**:
  - kernel_size <= 0 (ValueError)
  - stride <= 0 (ValueError)
  - padding < 0 或 > kernel_size/2 (ValueError)
  - dilation <= 0 (ValueError)
  - 维度不匹配 (RuntimeError)
  - output_ratio 超出 [0,1] 范围 (ValueError)

## 4. 覆盖映射
| TC ID | 需求覆盖 | 约束覆盖 | 风险点 |
|-------|----------|----------|--------|
| TC-01 | 基本池化实例化、前向传播 | kernel_size>0, stride>0, padding约束 | 类型别名定义 |
| TC-02 | 平均池化计算验证 | 输出形状公式 | 浮点计算误差 |
| TC-05 | 自适应池化输出控制 | output_size 参数 | 边界尺寸处理 |
| TC-08 | 分数池化随机性 | output_size/output_ratio | 随机性控制机制 |
| TC-03 | return_indices 功能 | 元组返回结构 | 索引正确性 |
| TC-04 | ceil_mode 边界 | 形状计算逻辑 | 窗口超出边界 |
| TC-07 | 参数异常验证 | 所有参数约束 | 错误消息完整性 |

**尚未覆盖的关键风险点**:
- dilation 参数的最大值约束未明确
- 分数最大池化的随机性控制机制细节
- 自适应池化中 output_size 的边界处理逻辑
- 不同设备（CUDA）的一致性验证
- LPPool 的 p-norm 计算精度验证