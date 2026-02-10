# tensorflow.python.data.experimental.ops.interleave_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock 弃用警告，fixture 管理数据集生命周期
- 随机性处理：固定随机种子，控制 sloppy 参数影响
- 弃用处理：验证 warnings.warn 被正确调用

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04, CASE_05
- **DEFERRED_SET**: CASE_06, CASE_07, CASE_08
- **group 列表**:
  - G1: parallel_interleave 函数族（核心函数族）
  - G2: 采样与选择函数族（采样与选择函数族）
- **active_group_order**: G1, G2
- **断言分级策略**: 首轮使用 weak 断言，最终轮启用 strong 断言
- **预算策略**: 
  - size: S（小型测试）
  - max_lines: 60-80 行
  - max_params: 4-6 个参数
  - 所有用例支持参数化

## 3. 数据与边界
- **正常数据集**: 简单 range 数据集，模拟 TFRecord 数据集
- **随机生成策略**: 固定种子确保可重现性
- **边界值**:
  - cycle_length=1 最小有效值
  - block_length=1 最小有效值
  - 单数据集列表边界
  - 空权重列表（默认均匀分布）
- **极端形状**: 大规模数据集（性能边界）
- **空输入**: 空 datasets 列表触发异常
- **负例与异常场景**:
  - cycle_length ≤ 0 触发 ValueError
  - weights 长度不匹配触发 ValueError
  - choice_dataset 值越界触发 InvalidArgumentError
  - 非 Dataset 类型参数触发 TypeError

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖约束 | 风险点 |
|-------|----------|----------|--------|
| TC-01 | parallel_interleave 基本功能 | 弃用警告、返回可调用对象、数据集结构 | map_func 类型约束不明确 |
| TC-02 | parallel_interleave 参数边界 | sloppy 参数、最小 cycle_length | 未明确 cycle_length 最大值 |
| TC-03 | sample_from_datasets_v2 基本采样 | 权重采样、随机种子、分布验证 | 权重参数类型转换细节 |
| TC-04 | choose_from_datasets_v2 基本选择 | 选择逻辑、choice_dataset 验证 | 空数据集检测时机 |
| TC-05 | parallel_interleave 异常处理 | 参数验证、错误消息 | 错误恢复行为未定义 |

**尚未覆盖的关键风险点**:
- buffer_output_elements/prefetch_input_elements 默认值未说明
- 复杂嵌套数据集结构兼容性
- 多设备环境兼容性
- 大规模数据集性能边界
- 权重归一化浮点精度误差