# torch.autograd.profiler 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 fixtures 隔离测试环境，mock CUDA 相关调用
- 随机性处理：固定随机种子，控制张量生成
- 线程安全：验证分析器的线程本地行为
- 状态管理：确保分析器状态正确恢复

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基本 CPU 分析）、CASE_02（禁用分析器）、CASE_03（形状记录功能）
- **DEFERRED_SET**: CASE_04（嵌套调用检测）、CASE_05（内存分析功能）
- **group 列表**:
  - G1: 基础功能测试（CASE_01, CASE_02, CASE_04）
  - G2: 高级配置测试（CASE_03, CASE_05）
  - G3: 边界与异常测试（deferred）
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮使用 weak 断言，最终轮启用 strong 断言
- **预算策略**: 
  - size: S（小型测试，70行以内）
  - max_lines: 60-85行
  - max_params: 6个参数
  - is_parametrized: 否（首轮简化）

## 3. 数据与边界
- **正常数据集**: 随机生成浮点张量，形状为 [2, 3, 4] 等小尺寸
- **随机生成策略**: 使用固定种子，生成正态分布数据
- **边界值**:
  - 空操作序列（无 autograd 操作）
  - 极端形状张量（0维、超大形状）
  - 混合精度操作（float16, float32, float64）
  - 特殊数值（inf, nan, 极大/极小值）
- **负例与异常场景**:
  - 嵌套 profile 调用（应抛出 RuntimeError）
  - use_cuda=True 但无 CUDA 设备
  - 无效 experimental_config 类型
  - 异步任务中的分析器传播

## 4. 覆盖映射
- **TC-01 → 需求**: 基本 CPU 分析功能，验证上下文管理器工作
- **TC-02 → 需求**: 禁用分析器时无事件记录
- **TC-03 → 需求**: 形状记录功能正确性
- **TC-04 → 需求**: 非可重入性约束验证
- **TC-05 → 需求**: 内存分析功能验证

- **尚未覆盖的风险点**:
  - CUDA 分析（需要 CUDA 设备）
  - FLOPs 估计准确性
  - 源代码归属功能
  - Kineto 集成测试
  - 实验性配置选项
  - 异步事件处理逻辑
  - 多线程环境行为
  - 导出文件格式验证

## 5. 迭代策略
- **首轮 (round1)**: 仅生成 SMOKE_SET 中的 3 个核心用例，使用 weak 断言
- **后续轮 (roundN)**: 修复失败用例，逐步添加 DEFERRED_SET 用例
- **最终轮 (final)**: 启用 strong 断言，可选覆盖扩展

## 6. 文件组织
- 主测试文件: `tests/test_torch_autograd_profiler.py`
- 分组文件:
  - G1: `tests/test_torch_autograd_profiler_basic.py`
  - G2: `tests/test_torch_autograd_profiler_advanced.py`
  - G3: `tests/test_torch_autograd_profiler_edge.py`