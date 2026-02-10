# tensorflow.python.ops.sparse_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（针对底层gen_sparse_ops）
- 随机性处理：固定随机种子，确定性测试数据生成
- 执行模式：支持eager和graph模式验证

## 2. 生成规格摘要（来自test_plan.json）
- **SMOKE_SET**: CASE_01 (from_dense基本转换), CASE_02 (sparse_to_dense基本填充), CASE_03 (稀疏矩阵乘法)
- **DEFERRED_SET**: CASE_04 (边界条件-空稀疏张量), CASE_05 (异常输入验证)
- **测试文件路径**: tests/test_tensorflow_python_ops_sparse_ops.py（单文件）
- **断言分级策略**: 首轮使用weak断言（形状、类型、基本属性），后续启用strong断言（数值精度、性能）
- **预算策略**: 
  - Size S: 65-75行，5-6参数
  - Size M: 85行，7参数
  - 所有用例参数化，减少重复代码

## 3. 数据与边界
- **正常数据集**: 小规模稀疏张量（1D/2D），非零值随机分布
- **随机生成策略**: 固定种子，控制稀疏度（10-30%）
- **边界值**:
  - 空稀疏张量（nnz=0）
  - 全零密集张量转换
  - 单元素稀疏张量
  - 极端稀疏度（nnz=1, shape大）
- **极端形状**:
  - 一维长向量（shape=[1000]）
  - 二维大矩阵（shape=[100,100]）
  - 高维张量（rank>2）
- **空输入**: 空索引、空值数组
- **负例与异常场景**:
  - 未排序索引触发InvalidArgumentError
  - 重复索引触发InvalidArgumentError
  - 维度不匹配触发ValueError
  - 负shape值触发InvalidArgumentError
  - 类型不匹配触发TypeError

## 4. 覆盖映射
| TC_ID | 需求覆盖 | 约束验证 | 优先级 |
|-------|----------|----------|--------|
| TC-01 | 密集到稀疏转换 | 索引排序、值提取 | High |
| TC-02 | 稀疏到密集转换 | 值填充、默认值处理 | High |
| TC-03 | 稀疏矩阵乘法 | 维度兼容、数学正确性 | High |
| TC-04 | 边界条件处理 | 空张量、极端形状 | High |
| TC-05 | 异常输入验证 | 错误类型、错误消息 | High |

### 尚未覆盖的风险点
- 跨设备（CPU/GPU）一致性验证
- 图模式与eager模式等价性
- 大尺寸张量性能基准
- 内存使用模式验证
- 浮点特殊值（inf, nan）处理

### 迭代策略
1. **首轮（Round1）**: 仅生成SMOKE_SET（3个核心用例），使用weak断言
2. **后续轮次（RoundN）**: 修复失败用例，每次最多处理3个block，提升deferred用例
3. **最终轮（Final）**: 启用strong断言，可选覆盖率目标

### Mock策略
- CASE_03需要mock底层gen_sparse_ops、framework.ops、eager.context
- 其他用例无需mock，直接测试公共API
- mock目标确保与requirements中的约束一致