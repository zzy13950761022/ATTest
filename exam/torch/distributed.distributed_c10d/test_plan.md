# torch.distributed.distributed_c10d 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock进程组、模拟分布式环境、使用fixtures管理测试状态
- 随机性处理：固定随机种子生成测试张量，控制RNG确保可重复性
- 环境模拟：使用monkeypatch模拟后端检测和网络通信

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04, CASE_05
- **DEFERRED_SET**: CASE_06, CASE_07, CASE_08, CASE_09, CASE_10, CASE_11, CASE_12
- **group列表**: G1(进程组管理), G2(集体通信), G3(点对点通信)
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮使用weak断言，最终轮启用strong断言
- **预算策略**: 
  - size: S(小型)或M(中型)
  - max_lines: 60-85行
  - max_params: 5-6个参数
  - 所有用例都需要mock支持

## 3. 数据与边界
- **正常数据集**: 小规模张量(2x2, 3x3, 4维)，float32/float64类型，CPU设备
- **随机生成策略**: 固定种子生成随机张量，确保可重复测试
- **边界值**: 
  - world_size=0或负数
  - rank超出有效范围
  - 空张量或零维张量
  - 超时值为0或负数
  - 极端大张量(内存边界)
- **负例与异常场景**:
  - 无效backend字符串
  - 张量类型不匹配
  - 进程组未初始化
  - init_method与store冲突
  - 异步操作超时

## 4. 覆盖映射
| TC_ID | 对应需求 | 覆盖约束 | 优先级 |
|-------|----------|----------|--------|
| TC-01 | 进程组初始化 | 基本功能验证 | High |
| TC-02 | 错误处理 | 无效参数异常 | High |
| TC-03 | all_reduce | SUM操作正确性 | High |
| TC-04 | broadcast | 数据广播功能 | High |
| TC-05 | 点对点通信 | 异步发送接收 | High |

**尚未覆盖的风险点**:
- NCCL后端GPU独占访问
- MPI编译依赖
- 多机网络通信
- 复数张量所有操作类型
- 超时机制完整验证
- 性能基准测试