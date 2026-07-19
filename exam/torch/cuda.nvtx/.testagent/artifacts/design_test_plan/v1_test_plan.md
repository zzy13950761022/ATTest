# torch.cuda.nvtx 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock CUDA环境、_nvtx模块调用、跨线程同步
- 随机性处理：固定随机种子、控制字符串生成

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04
- DEFERRED_SET: CASE_05, CASE_06, CASE_07
- group列表：G1(基础函数), G2(跨线程范围), G3(上下文管理器)
- active_group_order: G1 → G2 → G3
- 断言分级策略：首轮仅使用weak断言，最终轮启用strong断言
- 预算策略：size=S, max_lines=50-70, max_params=3-6

## 3. 数据与边界
- 正常数据集：ASCII字符串消息，简单嵌套深度
- 边界值：空字符串、长ASCII字符串、多级嵌套(3层)
- 极端形状：大量并发范围(5个)、格式化参数
- 空输入：空字符串消息
- 负例：非ASCII字符串、无效range_id、CUDA不可用
- 异常场景：类型错误、范围栈不平衡、跨线程同步问题

## 4. 覆盖映射
- TC-01: 覆盖range_push/range_pop嵌套栈功能
- TC-02: 覆盖mark事件标记功能  
- TC-03: 覆盖range_start/range_end跨线程管理
- TC-04: 覆盖range上下文管理器
- TC-05: 覆盖CUDA不可用异常处理

## 5. 尚未覆盖的风险点
- mark函数返回值不明确
- 跨线程范围跟踪的具体限制
- 非ASCII字符串的具体处理方式
- 性能影响和线程安全性
- 大量嵌套范围的性能表现