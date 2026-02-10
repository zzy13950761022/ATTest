# torch._linalg_utils 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 设备隔离：CPU与CUDA分别测试，CUDA不可用时跳过

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_05, CASE_06, CASE_09
- DEFERRED_SET: CASE_03, CASE_04, CASE_07, CASE_08, CASE_10, CASE_11, CASE_12
- group 列表与 active_group_order: G1, G2, G3
- 断言分级策略：首轮使用weak断言，最终轮启用strong断言
- 预算策略：每个CASE限制80行代码，最多6个参数

## 3. 数据与边界
- 正常数据集：随机生成浮点矩阵，固定随机种子保证可重复性
- 边界值：空矩阵、零维度、极大/极小形状（1×1到100×100）
- 极端数值：NaN、Inf、极大/极小浮点值
- 稀疏矩阵：COO格式稀疏矩阵，不同稀疏度测试
- 设备差异：CPU与CUDA实现路径分别验证

## 4. 覆盖映射
- G1组：矩阵运算核心函数（matmul, bform, qform）
- G2组：特征值与正交基函数（symeig, basis）
- G3组：辅助函数与异常处理（conjugate, transpose等）

## 5. 风险点
- 稀疏矩阵格式限制未明确说明
- CUDA设备可用性影响测试执行
- 已弃用函数的错误消息格式可能变化
- 部分函数缺少完整docstring，行为推断存在风险
- 内存使用边界未定义，大规模矩阵可能OOM