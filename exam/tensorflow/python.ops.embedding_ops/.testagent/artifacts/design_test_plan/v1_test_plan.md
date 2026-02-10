# tensorflow.python.ops.embedding_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 设备隔离：CPU优先，GPU可选
- 精度处理：不同dtype使用相应容差

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03
- **DEFERRED_SET**: CASE_04, CASE_05
- **测试文件路径**: tests/test_tensorflow_python_ops_embedding_ops.py
- **断言分级策略**: 首轮weak断言，最终轮启用strong断言
- **预算策略**: 
  - S类用例: max_lines=70-75, max_params=5-6
  - M类用例: max_lines=80-85, max_params=7
- **迭代策略**:
  - Round1: 仅SMOKE_SET，weak断言，最多5个用例
  - RoundN: 修复失败用例，提升deferred用例
  - Final: 启用strong断言，可选覆盖率

## 3. 数据与边界
- **正常数据集**: 小规模随机嵌入矩阵，形状[5-10, 2-8]
- **随机生成策略**: 固定种子，均匀分布随机值
- **边界值**: 
  - vocab_size=0或1的极端情况
  - embedding_dim=0或1的最小维度
  - 空ids张量或无值稀疏张量
  - 权重全0或负值的特殊场景
  - 超出vocab_size的无效ID
- **负例与异常场景**:
  - 参数类型错误
  - 形状不匹配
  - 无效组合器字符串
  - 负的max_norm值
  - 未排序稀疏索引

## 4. 覆盖映射
| TC_ID | 覆盖需求 | 关键约束 | 优先级 |
|-------|----------|----------|--------|
| TC-01 | 密集查找基本功能 | 形状匹配，值正确 | High |
| TC-02 | 稀疏查找组合器 | 三种组合器逻辑 | High |
| TC-03 | L2范数裁剪 | 裁剪边界条件 | High |
| TC-04 | 分区策略差异 | mod vs div策略 | High |
| TC-05 | 安全查找默认ID | 无效ID处理 | High |

## 5. 尚未覆盖的风险点
- RaggedTensor输入支持
- 多设备(CPU/GPU)一致性
- 不同数值精度全面验证
- 大规模词汇表性能
- 梯度计算正确性
- 动态形状支持
- 嵌套分区策略

## 6. Mock目标
- tensorflow.python.ops.sparse_ops.sparse_segment_*
- tensorflow.python.ops.math_ops._clip_by_norm
- tensorflow.python.ops.array_ops.gather
- tensorflow.python.framework.ops.colocate_with

## 7. 验证策略
- **弱断言**: 形状、dtype、无NaN/Inf、基本逻辑
- **强断言**: 精确值匹配、梯度正确性、设备一致性
- **Oracle**: 手动计算验证、参考实现对比
- **容差**: float32(1e-5), float16(1e-2), 组合操作数值稳定性