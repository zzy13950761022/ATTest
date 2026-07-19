# tensorflow.python.ops.ragged.ragged_functional_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用 pytest fixtures 管理 TensorFlow 会话和资源
- 随机性处理：固定随机种子确保测试可重复性
- 设备隔离：优先使用 CPU 设备，避免 GPU 依赖

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03
- **DEFERRED_SET**: CASE_04, CASE_05
- **测试文件路径**: tests/test_tensorflow_python_ops_ragged_ragged_functional_ops.py
- **断言分级策略**: 首轮使用 weak 断言（形状、类型、基本属性），后续启用 strong 断言（近似相等、精确值、嵌套行分割）
- **预算策略**: 每个用例 size=S，max_lines=80，max_params=6，全部参数化

## 3. 数据与边界
- **正常数据集**: 使用 tf.ragged.constant 创建典型 RaggedTensor（如 [[1,2,3], [], [4,5], [6]]）
- **随机生成策略**: 固定种子生成随机形状和值的 RaggedTensor
- **边界值**: 空 RaggedTensor、全空行、单元素、极端 ragged_rank
- **极端形状**: 超大 flat_values 长度、深度嵌套结构
- **空输入**: tf.ragged.constant([]) 和 tf.ragged.constant([[], []])
- **负例与异常场景**:
  - op 不是可调用对象
  - 多个 RaggedTensor nested_row_splits 不同
  - op 返回值 shape[0] 不匹配
  - partition dtypes 不兼容
  - 嵌套数据结构深度超限

## 4. 覆盖映射
| TC_ID | 需求/约束覆盖 | 优先级 |
|-------|--------------|--------|
| TC-01 | 单个 RaggedTensor 输入，简单 op | High |
| TC-02 | 多个 RaggedTensor 输入，相同 nested_row_splits | High |
| TC-03 | 无 RaggedTensor 输入，直接调用 op | High |
| TC-04 | RaggedTensor 在嵌套结构（列表）中 | High |
| TC-05 | op 返回值 shape[0] 不匹配的错误处理 | High |

## 5. 尚未覆盖的风险点
- 复杂嵌套数据结构（字典、元组混合）
- 不同 partition dtypes 的自动转换行为
- 大规模数据性能边界
- op 函数签名验证的完整性
- 内存使用和资源泄漏风险

## 6. 迭代策略
- **首轮**: 仅生成 SMOKE_SET（3个核心用例），使用 weak 断言
- **后续轮次**: 修复失败用例，逐步启用 deferred_set
- **最终轮次**: 启用 strong 断言，可选覆盖率检查

## 7. 依赖管理
- 无需 mock 核心函数（首轮）
- 后续可能需要 mock 辅助函数以隔离测试
- 设备要求：优先 CPU，避免测试环境差异