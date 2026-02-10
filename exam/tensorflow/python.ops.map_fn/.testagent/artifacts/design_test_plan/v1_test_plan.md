# tensorflow.python.ops.map_fn 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
  - 使用mock隔离autograph、while_loop、nest等内部依赖
  - 使用fixtures管理测试数据和资源
  - 使用monkeypatch控制环境变量和全局状态
- 随机性处理：固定随机种子/控制RNG
  - 使用固定随机种子确保测试可重复性
  - 控制TensorFlow随机数生成器状态

## 2. 生成规格摘要（来自test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_05
- DEFERRED_SET: CASE_03, CASE_04
- 测试文件路径：tests/test_tensorflow_python_ops_map_fn.py（单文件）
- 断言分级策略：
  - 首轮使用weak断言：形状匹配、类型匹配、基本功能
  - 后续启用strong断言：精确值、梯度检查、性能边界
- 预算策略：
  - S级用例：max_lines 65-75, max_params 7-8
  - M级用例：max_lines 85-90, max_params 7-8
  - 所有用例均支持参数化

## 3. 数据与边界
- 正常数据集与随机生成策略：
  - 使用固定种子生成随机张量
  - 覆盖常见dtype：float32, float64, int32, int64
  - 形状范围：[1,10]到[5,4,5]维度
- 边界值/极端形状/空输入：
  - 零维张量（标量）输入
  - 空张量（shape包含0）
  - 单元素张量边界
  - 大维度内存边界测试
- 负例与异常场景列表：
  - elems为空序列
  - 嵌套张量外维度不一致
  - fn签名不同未指定输出签名
  - parallel_iterations<=0
  - 使用已弃用dtype参数

## 4. 覆盖映射
- 每个TC对应的需求/约束：

| TC-ID | 覆盖需求 | 关键约束 |
|-------|----------|----------|
| TC-01 | 基本单张量映射 | 形状关系：[elems.shape[0]] + fn(elems[0]).shape |
| TC-02 | 嵌套张量输入 | 嵌套张量相同外维度大小 |
| TC-03 | RaggedTensor处理 | fn接收每行数据，ragged_rank处理 |
| TC-04 | SparseTensor处理 | fn接收每行数据（维度减1） |
| TC-05 | 输出签名要求 | fn输入输出签名不同时必须指定fn_output_signature |

- 尚未覆盖的风险点：
  - eager模式下parallel_iterations>1不会真正并行执行
  - dtype参数已弃用但仍有代码路径
  - 大尺寸张量性能边界
  - 极端数值处理（inf, nan）
  - 不同设备（CPU/GPU）行为差异