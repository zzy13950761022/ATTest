# torch.autograd.grad_mode 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用fixtures管理张量创建，mock torch.set_grad_enabled调用
- 随机性处理：固定随机种子确保可重复性
- 设备支持：优先CPU，参数化扩展GPU

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04（4个核心用例）
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07（3个延期用例）
- **group列表**: G1（基础梯度控制类）, G2（高级模式与装饰器）
- **active_group_order**: G1 → G2（按依赖顺序）
- **断言分级策略**: 首轮仅使用weak断言（状态验证、无异常）
- **预算策略**: 所有用例size=S，max_lines≤80，max_params≤6

## 3. 数据与边界
- **正常数据集**: float32/float64张量，CPU设备，requires_grad=True/False
- **边界值**: 空上下文、嵌套上下文、装饰器包装
- **极端形状**: 不适用（主要测试状态管理）
- **负例场景**: 
  - 非法mode参数类型
  - 异常退出状态恢复
  - 装饰器异常传播

## 4. 覆盖映射
| TC ID | 需求覆盖 | 约束覆盖 |
|-------|----------|----------|
| TC-01 | no_grad基础功能 | requires_grad=False，状态恢复 |
| TC-02 | enable_grad交互 | 嵌套上下文状态管理 |
| TC-03 | inference_mode | mode参数控制，视图跟踪禁用 |
| TC-04 | set_grad_enabled | 必需参数验证，状态设置 |
| TC-05 | 装饰器用法 | 函数包装，梯度状态正确 |

## 5. 尚未覆盖的风险点
- inference_mode源码不完整（视图跟踪细节）
- 前向模式自动微分限制
- 多线程并发访问安全性
- 内存使用量变化量化验证
- torch.jit兼容性测试

## 6. 迭代策略
- **首轮**: 仅生成SMOKE_SET用例，使用weak断言
- **后续**: 修复失败用例，提升DEFERRED_SET优先级
- **最终**: 启用strong断言，可选覆盖率检查