# torch.nn.modules.upsampling 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 fixtures 管理测试数据和模块实例
- 随机性处理：固定随机种子确保可重现性
- 设备隔离：分别测试 CPU 和 CUDA（如可用）

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04, CASE_09（5个核心用例）
- **DEFERRED_SET**: CASE_05-08, CASE_10-12（7个延期用例）
- **group 列表**:
  - G1: Upsample 核心功能（CASE_01-02, 05-06）
  - G2: 专用子类功能（CASE_03-04, 07-08）
  - G3: 边界与异常处理（CASE_09-12）
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮使用 weak 断言（形状、类型、无NaN、基础功能），后续启用 strong 断言（精确匹配、梯度检查、内存泄漏）
- **预算策略**: 每个用例 size=S，max_lines=60-75，max_params=4-6

## 3. 数据与边界
- **正常数据集**: 随机生成标准形状张量（3D/4D/5D），固定随机种子
- **边界值**: scale_factor=1.0（恒等变换）、极小尺寸输入（1×1）、极大缩放因子
- **极端形状**: 单元素张量、非对称尺寸、批量不同尺寸
- **空输入**: 零尺寸张量（应触发错误）
- **负例场景**: 同时指定 size 和 scale_factor、无效 mode 值、负缩放因子、align_corners 用于非线性模式

## 4. 覆盖映射
| TC ID | 需求/约束覆盖 | 风险点 |
|-------|--------------|--------|
| TC-01 | Upsample size 参数功能 | 尺寸参数类型转换 |
| TC-02 | Upsample scale_factor 参数功能 | align_corners 行为一致性 |
| TC-03 | UpsamplingNearest2d 基础功能 | 模式固定验证 |
| TC-04 | UpsamplingBilinear2d 基础功能 | align_corners=True 默认值 |
| TC-09 | 参数互斥性验证 | 错误消息准确性 |
| TC-05 | 多维度支持（延期） | 3D/5D 数据边界 |
| TC-06 | 多种插值模式（延期） | bicubic 精度容差 |
| TC-10 | 无效参数处理（延期） | 错误类型完整性 |

**尚未覆盖的关键风险点**:
- recompute_scale_factor 参数行为
- CUDA 与 CPU 结果一致性
- 内存不足时的错误处理
- 梯度计算正确性（训练模式）
- 多线程环境行为