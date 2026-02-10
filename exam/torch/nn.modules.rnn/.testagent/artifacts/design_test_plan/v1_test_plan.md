# torch.nn.modules.rnn 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 fixtures 管理 RNN 实例，mock 随机数生成器用于 dropout 测试
- 随机性处理：固定随机种子保证可重复性，控制 RNG 状态
- 设备管理：优先 CPU 测试，CUDA 作为可选扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基础RNN）、CASE_02（LSTM）、CASE_03（双向GRU）、CASE_04（多层dropout）
- **DEFERRED_SET**: CASE_05（LSTM投影）、CASE_06-CASE_08（待定义）
- **group 列表**:
  - G1: 核心RNN/LSTM/GRU正向传播（CASE_01,02,05,06）
  - G2: 高级功能与边界条件（CASE_03,04,07,08）
- **active_group_order**: ["G1", "G2"]
- **断言分级策略**: 首轮仅使用 weak 断言（形状、类型、有限性检查），最终轮启用 strong 断言（近似相等、梯度检查）
- **预算策略**: 
  - S 尺寸：max_lines=70-85, max_params=8-9
  - M 尺寸：max_lines=80-85, max_params=9
  - 所有用例均参数化，仅 CASE_04 需要 mock

## 3. 数据与边界
- **正常数据集**: 随机生成小规模张量（batch_size≤4, seq_len≤10, hidden_size≤40）
- **边界值测试**:
  - 单层/多层网络边界
  - dropout=0.0/1.0 边界
  - batch_first=True/False 格式转换
  - 双向RNN输出维度验证
- **极端形状**:
  - batch_size=1, seq_len=1 最小输入
  - 单层双向RNN特殊处理
  - 投影尺寸约束（proj_size < hidden_size）
- **空输入与异常**:
  - 无效mode字符串异常
  - 形状不匹配异常
  - 非LSTM使用proj_size异常
  - dropout在单层网络警告

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖约束 | 风险点 |
|-------|----------|----------|--------|
| TC-01 | 基础正向传播 | 形状正确性、类型检查 | 浮点精度差异 |
| TC-02 | LSTM功能 | 隐藏/细胞状态管理 | 门控单元复杂性 |
| TC-03 | 双向RNN | 输出维度2×hidden_size | 前后向参数共享 |
| TC-04 | dropout随机性 | 多层网络dropout有效性 | 随机种子管理 |
| TC-05 | LSTM投影 | proj_size约束检查 | 投影权重初始化 |

**尚未覆盖的关键风险点**:
- PackedSequence输入处理
- CUDA/cuDNN特定优化路径
- 内存碎片化影响
- 并行计算线程安全性
- 单元版本（RNNCell/LSTMCell/GRUCell）独立测试

## 5. 迭代策略
1. **首轮（Round1）**: 仅生成 SMOKE_SET 中的4个核心用例，使用weak断言
2. **中间轮（RoundN）**: 修复失败用例，每次最多处理3个block，提升deferred用例
3. **最终轮（Final）**: 启用strong断言，可选覆盖率检查，补齐所有deferred用例