# torch.nn.modules.linear 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 fixtures 管理模型实例，mock 控制随机初始化
- 随机性处理：固定随机种子确保可重复性，控制 RNG 状态
- 设备管理：优先 CPU 测试，CUDA 作为可选扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (Linear基础), CASE_02 (无偏置), CASE_05 (Bilinear), CASE_08 (LazyLinear)
- **DEFERRED_SET**: CASE_03, CASE_04, CASE_06, CASE_07, CASE_09, CASE_10
- **group 列表**: 
  - G1: Linear 核心功能 (CASE_01-04)
  - G2: 特殊线性层 (CASE_05-07)  
  - G3: 延迟初始化与边界 (CASE_08-10)
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮仅 weak 断言（形状、类型、有限性检查），最终轮启用 strong 断言（数值精度、梯度检查）
- **预算策略**: 
  - S 大小用例: max_lines=60-80, max_params=4-6
  - M 大小用例: max_lines=85-90, max_params=6-8

## 3. 数据与边界
- **正常数据集**: 随机生成符合形状的浮点张量，固定种子确保可重复
- **边界值**: 
  - 最小维度 (in_features=1, out_features=1)
  - 大维度内存检查 (1000×500)
  - 多批次维度 (3D+ 输入)
  - 空批次 (*=0) 边缘情况
- **极端形状**: 超大维度测试内存，超小维度测试边界
- **负例与异常场景**:
  - 输入维度不匹配触发 RuntimeError
  - 非法维度值触发 ValueError  
  - 非数值类型输入触发 TypeError
  - LazyLinear 重复初始化保护

## 4. 覆盖映射
| TC ID | 需求/约束覆盖 | 优先级 |
|-------|--------------|--------|
| TC-01 | Linear 基础正向传播验证 | High |
| TC-02 | bias=False 配置验证 | High |
| TC-03 | 不同 dtype 精度验证 | Medium |
| TC-04 | 不同输入形状支持 | Medium |
| TC-05 | Bilinear 双输入变换 | High |
| TC-06 | Identity 恒等映射 | Medium |
| TC-07 | Bilinear 无偏置配置 | Medium |
| TC-08 | LazyLinear 延迟初始化 | High |
| TC-09 | 异常输入处理 | Medium |
| TC-10 | 初始化算法验证 | Medium |

## 5. 尚未覆盖的风险点
- in_features=0 或 out_features=0 的边界行为未定义
- TensorFloat32 和 ROCm float16 的特殊精度处理
- 量化相关类 NonDynamicallyQuantizableLinear 的特殊用途
- 多线程环境下的并发安全性
- 极端数值 (inf, nan) 的传播行为验证

## 6. 迭代策略
- **首轮 (round1)**: 仅生成 SMOKE_SET (4个用例)，使用 weak 断言
- **后续轮 (roundN)**: 修复失败用例，提升 DEFERRED_SET，每次最多3个新用例
- **最终轮 (final)**: 启用 strong 断言，可选覆盖率检查

## 7. 文件组织
- 主文件: `tests/test_torch_nn_modules_linear.py`
- 分组文件: `test_torch_nn_modules_linear_g1.py` (G1), `_g2.py` (G2), `_g3.py` (G3)
- 所有模式: `test_torch_nn_modules_linear_*.py`