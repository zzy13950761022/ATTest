# tensorflow.python.ops.parallel_for.control_flow_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures 控制依赖
- 随机性处理：固定随机种子，控制 RNG 状态
- 设备隔离：CPU-only 优先，GPU 可选扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01 (for_loop基础功能), CASE_02 (pfor向量化转换), CASE_03 (vectorized_map广播功能)
- **DEFERRED_SET**: CASE_04 (CompositeTensor支持), CASE_05 (fallback_to_while_loop机制)
- **测试文件路径**: tests/test_tensorflow_python_ops_parallel_for_control_flow_ops.py (单文件)
- **断言分级策略**: 首轮使用 weak 断言，最终轮启用 strong 断言
- **预算策略**: 
  - Size: S (小型测试，70-90行)
  - max_lines: 70-90行
  - max_params: 5-6个参数
  - is_parametrized: true (支持参数化扩展)

## 3. 数据与边界
- **正常数据集**: 随机生成浮点/整数张量，形状 [2-10, 2-10]
- **边界值**: 
  - iters=0 (零迭代返回空结构)
  - iters=1 (单次迭代边界)
  - parallel_iterations=null (默认并行)
  - 空形状 [0,0] 张量
- **极端形状**:
  - 高瘦形状 [1, 100]
  - 宽扁形状 [100, 1]
  - 大尺寸 [10, 10] 迭代
- **负例与异常场景**:
  - iters 负数触发 ValueError
  - parallel_iterations <= 0 错误
  - pfor 中 parallel_iterations=1 错误
  - 不支持的控制流操作
  - 形状依赖输入的 loop_fn

## 4. 覆盖映射
| TC ID | 对应需求 | 关键约束 | 风险点 |
|-------|----------|----------|--------|
| TC-01 | for_loop基础功能 | iters非负，形状独立 | 内存管理，并行控制 |
| TC-02 | pfor向量化转换 | parallel_iterations>1 | 向量化失败，回退机制 |
| TC-03 | vectorized_map广播 | 沿第一维展开 | 广播语义，梯度计算 |
| TC-04 | CompositeTensor支持 | SparseTensor/IndexedSlices | 稀疏性保持，重组正确性 |
| TC-05 | fallback机制 | fallback_to_while_loop | 性能退化，错误传播 |

**尚未覆盖的关键风险点**:
- XLA 编译上下文行为差异
- 跨设备 (CPU/GPU) 一致性
- 内存使用边界和泄漏
- 复杂嵌套结构处理
- 实验性 API 变更兼容性