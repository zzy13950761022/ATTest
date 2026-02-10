# torch.utils.checkpoint 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 设备支持：CPU + CUDA（可选）
- 梯度验证：与直接计算对比

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基础功能）, CASE_02（参数模式）, CASE_03（梯度验证）
- **DEFERRED_SET**: CASE_04（异常处理）, CASE_05（RNG管理）, CASE_06（嵌套结构）
- **group 列表**:
  - G1: 基础功能验证（CASE_01, CASE_02, CASE_04）
  - G2: 梯度与RNG验证（CASE_03, CASE_05, CASE_06）
- **active_group_order**: ["G1", "G2"]
- **断言分级策略**: 首轮使用weak断言，最终轮启用strong断言
- **预算策略**: 
  - size: S/M（70-85行）
  - max_lines: 65-85行
  - max_params: 4-6个参数
  - 所有CASE都支持参数化

## 3. 数据与边界
- **正常数据集**: 简单线性函数、随机操作函数、嵌套输出函数
- **随机生成策略**: 固定随机种子确保可重复性
- **边界值测试**:
  - 空参数列表 function()
  - 0维Tensor输入
  - 大尺寸Tensor内存边界
  - 混合设备输入(CPU+CUDA)
  - requires_grad=False的Tensor
- **负例与异常场景**:
  - function不可调用（TypeError）
  - use_reentrant=True时传递kwargs（RuntimeError）
  - 前向/反向行为不一致（RuntimeError）
  - 嵌套Tensor结构处理异常

## 4. 覆盖映射
- **TC-01**: 基础功能验证 → 需求1.1, 2.1, 3.1
- **TC-02**: use_reentrant模式 → 需求2.3, 4.1
- **TC-03**: 梯度正确性 → 需求3.2, 6.1
- **TC-04**: 异常处理 → 需求4.1, 4.2
- **TC-05**: RNG状态管理 → 需求2.5, 3.3
- **TC-06**: 嵌套结构处理 → 需求3.1, 6.2

- **尚未覆盖的风险点**:
  - 多次嵌套checkpoint调用
  - 自定义autograd.Function作为function
  - 多线程/多进程环境
  - 性能开销量化标准
  - CUDA状态管理细节
  - 自定义对象序列化支持

## 5. 迭代策略
- **首轮**: 仅生成SMOKE_SET（3个核心用例），使用weak断言
- **后续轮**: 修复失败用例，从DEFERRED_SET提升用例
- **最终轮**: 启用strong断言，可选覆盖率检查

## 6. Mock目标
- CASE_04: CheckpointFunction.apply, _checkpoint_without_reentrant
- CASE_05: torch RNG状态管理函数
- 其他CASE: 无需mock，直接测试真实功能