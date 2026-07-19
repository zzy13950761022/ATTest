# torch.nn.utils.weight_norm 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（仅CASE_05需要mock内部WeightNorm.apply）
- 随机性处理：固定随机种子，控制权重初始化

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_06（4个核心用例）
- DEFERRED_SET: CASE_04, CASE_05, CASE_07, CASE_08, CASE_09（5个延期用例）
- group列表：G1（核心功能验证）、G2（边界与异常处理）
- active_group_order: G1, G2（按优先级顺序）
- 断言分级策略：首轮仅使用weak断言，最终轮启用strong断言
- 预算策略：size=S/M（小/中），max_lines=60-85，max_params=5-8

## 3. 数据与边界
- 正常数据：Linear层（20×40, 10×20, 15×25等标准形状）
- 随机生成：固定种子生成随机权重，确保可重复性
- 边界值：dim=None全局范数，最小形状1×1权重
- 极端形状：大尺寸100×200权重，Conv层多维权重
- 空输入：不适用（模块参数必须存在）
- 负例：无效模块类型，不存在参数名，无效dim类型
- 异常场景：TypeError（模块类型错误），AttributeError（参数不存在）

## 4. 覆盖映射
| TC ID | 需求/约束覆盖 | 优先级 |
|-------|--------------|--------|
| TC-01 | 对Linear层应用默认参数权重归一化 | High |
| TC-02 | 验证参数分解正确性：w ≈ g * v/||v|| | High |
| TC-03 | 测试dim=None在整个张量上计算范数 | High |
| TC-04 | 测试不同参数名称（非'weight'）的归一化 | High |
| TC-05 | 验证前向传播钩子正确触发权重重计算 | High |
| TC-06 | 非法输入触发的异常（无效模块类型） | High |
| TC-07 | 不存在的参数名称错误处理 | Medium |
| TC-08 | 无效dim参数类型错误处理 | Medium |
| TC-09 | 不同模块类型（Conv2d）的归一化 | Medium |

## 5. 尚未覆盖的风险点
- WeightNorm.apply内部实现细节未知
- dim参数类型注解与实际接受None值的不一致
- 梯度计算和反向传播验证
- 极端数值情况（NaN/Inf/零权重）
- 与其他nn.Module方法的兼容性
- 多设备测试（CUDA）的完整覆盖