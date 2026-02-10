# tensorflow.python.ops.array_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock底层C++操作（gen_array_ops.*）、tensor转换和形状推断函数
- 随机性处理：固定随机种子，使用确定性数据生成

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (reshape基本形状变换), CASE_02 (expand_dims维度插入), CASE_03 (concat张量连接), CASE_04 (stack张量堆叠)
- **DEFERRED_SET**: CASE_05 (reshape自动推断维度), CASE_06 (unstack张量解堆叠)
- **group列表**: G1 (核心形状操作组), G2 (张量组合操作组)
- **active_group_order**: G1 → G2
- **断言分级策略**: 首轮使用weak断言（形状、类型、基本属性），后续启用strong断言（精确值、内存布局）
- **预算策略**: 每个CASE size=S, max_lines≤80, max_params≤6

## 3. 数据与边界
- **正常数据集**: 使用numpy生成确定性测试数据，对比TensorFlow与NumPy结果
- **边界值**:
  - 空张量（shape=[]）和零维标量
  - 负轴索引边界（axis=-rank-1）
  - 形状参数包含-1（自动推断维度）
  - 极大形状参数（接近内存限制）
- **负例与异常场景**:
  - 无效形状参数（负值且不为-1）
  - 形状不兼容（reshape元素总数不匹配）
  - 越界axis参数
  - 空张量列表（concat, stack）
  - 不兼容的输入张量形状
  - 类型不匹配（非数值类型）

## 4. 覆盖映射
| TC ID | 对应需求约束 | 覆盖函数 |
|-------|-------------|----------|
| TC-01 | reshape基本形状变换 | reshape |
| TC-02 | expand_dims维度插入 | expand_dims |
| TC-03 | concat多张量连接 | concat |
| TC-04 | stack张量堆叠 | stack |
| TC-05 | reshape自动推断维度 | reshape |
| TC-06 | unstack张量解堆叠 | unstack |

**尚未覆盖的风险点**:
- 模块无`__all__`定义，公共API边界模糊
- v1/v2版本函数差异
- GPU/TPU设备特定行为
- 大张量性能退化点
- 内存使用峰值未文档化