# tensorflow.python.data.experimental.ops.grouping 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock 弃用警告和函数包装器，使用 fixtures 管理数据集
- 随机性处理：固定随机种子确保可重现测试
- 测试分组：按功能拆分为 3 个 group（G1-G3）

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_05, CASE_06, CASE_09（5个核心用例）
- **DEFERRED_SET**: CASE_03, CASE_04, CASE_07, CASE_08, CASE_10, CASE_11, CASE_12（7个延期用例）
- **group 列表**:
  - G1: group_by_reducer 核心功能（CASE_01-04）
  - G2: bucket_by_sequence_length 序列分桶（CASE_05-08）
  - G3: group_by_window 与异常处理（CASE_09-12）
- **active_group_order**: G1 → G2 → G3（按优先级顺序）
- **断言分级策略**: 首轮仅使用 weak 断言，最终轮启用 strong 断言
- **预算策略**: 
  - S size: max_lines=60-75, max_params=5-6
  - M size: max_lines=85-110, max_params=7-9
  - 参数化测试优先，减少代码重复

## 3. 数据与边界
- **正常数据集**: 小型数值数据集（10-20个元素），简单键函数（取模、提取字段）
- **随机生成策略**: 固定种子生成变长序列和嵌套结构
- **边界值**:
  - 空数据集输入验证
  - 零窗口大小边界
  - 极大序列长度边界
  - 桶边界列表为空
  - 负值边界检查
- **极端形状**: 深度嵌套字典结构，多维张量序列
- **负例与异常场景**:
  - 键函数返回错误类型（非 tf.int64）
  - 键函数返回非标量张量
  - bucket_batch_sizes 长度不匹配
  - window_size 和 window_size_func 同时提供
  - 无效的 bucket_boundaries（非递增）
  - 非数据集对象输入
  - 用户函数抛出异常

## 4. 覆盖映射
| TC ID | 对应需求/约束 | 优先级 | 覆盖点 |
|-------|--------------|--------|--------|
| TC-01 | group_by_reducer 基本功能 | High | 分组归约核心路径 |
| TC-02 | 参数验证异常路径 | High | 键函数类型验证 |
| TC-05 | bucket_by_sequence_length 基本功能 | High | 序列分桶和填充 |
| TC-06 | 参数验证异常路径 | High | 桶参数验证 |
| TC-09 | 弃用警告正确触发 | High | 已弃用API行为 |
| TC-03 | 空数据集处理 | Medium | 边界值处理 |
| TC-07 | 填充选项验证 | Medium | 高级功能覆盖 |
| TC-10 | 参数互斥验证 | Medium | API约束验证 |
| TC-11 | 无效数据集输入 | Medium | 类型安全 |
| TC-04 | 复杂嵌套结构 | Low | 扩展场景 |
| TC-08 | 边界序列处理 | Low | 极端情况 |
| TC-12 | 函数包装器错误 | Low | 错误传播 |

## 5. 尚未覆盖的风险点
- Reducer 类的详细使用和测试
- 多设备环境（GPU/TPU）兼容性
- 大规模数据集性能基准
- 内存泄漏和资源清理验证
- 与替代API（tf.data.Dataset方法）的行为一致性

## 6. 迭代策略
- **首轮（round1）**: 仅生成 SMOKE_SET（5个用例），使用 weak 断言
- **后续轮（roundN）**: 修复失败用例，每次最多处理3个block，提升延期用例
- **最终轮（final）**: 启用 strong 断言，可选覆盖率目标

## 7. 依赖与模拟
- **需要 mock**: 弃用警告捕获、StructuredFunctionWrapper 行为
- **需要 fixtures**: 测试数据集生成、Reducer 实例创建
- **环境要求**: TensorFlow 数据集API可用，Python 3.7+