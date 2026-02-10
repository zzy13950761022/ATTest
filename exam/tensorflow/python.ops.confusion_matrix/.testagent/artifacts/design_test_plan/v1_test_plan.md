# tensorflow.python.ops.confusion_matrix 测试计划

## 1. 测试策略
- **单元测试框架**：pytest
- **隔离策略**：使用 TensorFlow 测试工具（如 tf.test.TestCase），无外部 mock 需求
- **随机性处理**：固定随机种子，使用确定性测试数据
- **设备兼容性**：支持 CPU 和 GPU 设备，默认使用 CPU 进行测试

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基本混淆矩阵计算）、CASE_02（自动num_classes推断）、CASE_03（带权重计算）
- **DEFERRED_SET**: CASE_04（数据类型转换）、CASE_05（边界值处理-空输入）
- **单文件路径**: `tests/test_tensorflow_python_ops_confusion_matrix.py`
- **断言分级策略**: 首轮使用 weak 断言（shape、dtype、basic_property），后续启用 strong 断言（exact_values、approx_equal）
- **预算策略**: 每个用例 size=S，max_lines=80，max_params=6，全部参数化

## 3. 数据与边界
- **正常数据集**: 使用小型确定性向量（长度3-5），包含典型分类场景
- **随机生成策略**: 固定种子生成整数标签和预测值，确保可重现性
- **边界值处理**:
  - 空张量输入（长度为0）
  - 单类别分类（num_classes=1）
  - 大标签值自动推断
  - 浮点数权重计算
  - 不同数据类型转换
- **负例与异常场景列表**:
  - 维度不匹配异常
  - 负标签值异常
  - 标签值大于等于num_classes异常
  - 权重形状不匹配异常
  - 非张量输入类型异常

## 4. 覆盖映射
- **TC-01 (CASE_01)**: 基本混淆矩阵计算 → 验证标准分类场景正确性
- **TC-02 (CASE_02)**: 自动num_classes推断 → 测试未指定类别数时的自动计算
- **TC-03 (CASE_03)**: 带权重计算 → 验证权重参数对矩阵计数的影响
- **TC-04 (CASE_04)**: 数据类型转换 → 测试不同dtype参数的正确性
- **TC-05 (CASE_05)**: 边界值处理-空输入 → 验证空输入和零类别边界场景

- **尚未覆盖的风险点**:
  - 浮点数标签的自动转换行为
  - 稀疏张量输入支持情况
  - 大向量性能表现（长度 > 10000）
  - 多设备兼容性详细验证
  - 向后兼容性（confusion_matrix_v1）