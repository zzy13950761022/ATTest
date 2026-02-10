# torch.serialization 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock文件系统操作、CUDA设备检测、pickle模块
- 随机性处理：固定随机种子生成测试张量
- 临时文件管理：使用pytest临时目录fixture

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基本张量保存加载）、CASE_02（内存文件对象支持）、CASE_03（设备映射功能）、CASE_04（weights_only安全模式）
- **DEFERRED_SET**: CASE_05（存储共享关系保持）等5个用例
- **group列表**: G1（基础保存加载功能）、G2（高级功能与设备映射）、G3（边界与异常处理）
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮使用weak断言（形状、类型、基本数值），最终轮启用strong断言（存储共享、元数据完整）
- **预算策略**: size=S/M（60-75行），max_params=4-6，首轮只生成SMOKE_SET

## 3. 数据与边界
- **正常数据集**: 随机生成浮点/整数张量（形状2x3到10x10），固定随机种子
- **边界值**: 空张量、None对象、特殊数值（inf/nan）、极大张量（内存边界）
- **极端形状**: 0维标量、高维张量（4D+）、包含0的维度
- **负例与异常场景**:
  1. 无效文件路径（FileNotFoundError）
  2. 不支持的文件对象（AttributeError）
  3. 损坏的序列化文件（EOFError）
  4. weights_only模式下不安全对象（RuntimeError）
  5. 不兼容pickle模块（RuntimeError）

## 4. 覆盖映射
| TC ID | 需求/约束覆盖 | 优先级 |
|-------|--------------|--------|
| TC-01 | 基本张量保存加载流程 | High |
| TC-02 | 内存文件对象（BytesIO）支持 | High |
| TC-03 | map_location设备映射功能 | High |
| TC-04 | weights_only安全模式验证 | High |
| TC-05 | 存储共享关系保持 | Medium |

**尚未覆盖的风险点**:
- register_package函数（文档不完整）
- pickle_module参数类型变体
- map_location复杂类型（字典、可调用函数）
- 并发访问和文件锁
- 网络文件系统路径处理