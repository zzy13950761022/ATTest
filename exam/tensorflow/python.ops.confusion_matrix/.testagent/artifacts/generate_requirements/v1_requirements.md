# tensorflow.python.ops.confusion_matrix 测试需求

## 1. 目标与范围
- **主要功能与期望行为**：验证混淆矩阵计算正确性，包括基本分类、带权重计算、自动维度推断、数据类型转换、边界值处理等核心功能
- **不在范围内的内容**：混淆矩阵的可视化展示、性能基准测试、分布式计算场景、稀疏张量输入、自定义损失函数计算

## 2. 输入与约束
- **参数列表**：
  - labels: Tensor, 必需, 1-D 向量, 真实标签值
  - predictions: Tensor, 必需, 1-D 向量, 预测值
  - num_classes: int/None, 可选, 默认根据预测和标签的最大值计算
  - weights: Tensor/None, 可选, 形状需与 predictions 匹配
  - dtype: dtype, 可选, 默认 tf.int32
  - name: str/None, 可选, 操作作用域名称

- **有效取值范围/维度/设备要求**：
  - labels 和 predictions 必须是相同形状的 1-D 向量
  - 标签和预测值不能为负数
  - 当指定 num_classes 时，标签和预测值必须小于 num_classes
  - 权重张量形状必须与预测值匹配
  - 类标签从 0 开始

- **必需与可选组合**：
  - labels 和 predictions 为必需参数
  - num_classes、weights、dtype、name 为可选参数
  - weights 可选但需与 predictions 形状匹配

- **随机性/全局状态要求**：
  - 无随机性要求
  - 不依赖全局状态

## 3. 输出与判定
- **期望返回结构及关键字段**：
  - 返回形状为 `[n, n]` 的二维张量，其中 n 是分类任务的可能标签数
  - 数据类型为指定的 dtype（默认 tf.int32）
  - 矩阵列表示预测标签，行表示真实标签

- **容差/误差界**：
  - 整数计算无容差要求
  - 浮点数权重计算需考虑浮点精度误差（相对误差 1e-6）

- **状态变化或副作用检查点**：
  - 无外部状态变化
  - 不修改输入张量
  - 不产生文件或网络操作

## 4. 错误与异常场景
- **非法输入/维度/类型触发的异常或警告**：
  - labels 和 predictions 维度不匹配
  - 标签或预测值为负数
  - 标签或预测值大于等于 num_classes（当指定时）
  - weights 形状与 predictions 不匹配
  - 非张量输入类型

- **边界值处理**：
  - 空张量输入（长度为0）
  - num_classes=None 时的自动推断
  - 单类别分类（num_classes=1）
  - 极端形状（大向量长度）
  - 极端数值（大标签值）

## 5. 依赖与环境
- **外部资源/设备/网络/文件依赖**：
  - 无外部资源依赖
  - 无需网络或文件访问
  - 支持 CPU 和 GPU 设备

- **需要 mock/monkeypatch 的部分**：
  - tensorflow.python.ops.array_ops.remove_squeezable_dimensions
  - tensorflow.python.ops.check_ops.assert_non_negative
  - tensorflow.python.ops.math_ops.cast
  - tensorflow.python.ops.math_ops.maximum
  - tensorflow.python.ops.array_ops.scatter_nd
  - tensorflow.python.ops.array_ops.zeros
  - tensorflow.python.ops.array_ops.expand_dims
  - tensorflow.python.ops.array_ops.reshape
  - tensorflow.python.ops.array_ops.size
  - tensorflow.python.ops.array_ops.range

## 6. 覆盖与优先级
- **必测路径（高优先级）**：
  1. 基本混淆矩阵计算：验证标准分类场景的正确性
  2. 自动 num_classes 推断：测试未指定类别数时的自动计算
  3. 带权重计算：验证权重参数对矩阵计数的影响
  4. 数据类型转换：测试不同 dtype 参数的正确性
  5. 边界值处理：验证空输入、单类别等边界场景

- **可选路径（中/低优先级）**：
  - 大向量性能测试（长度 > 10000）
  - 多设备兼容性（CPU/GPU）
  - 不同数值精度组合（int32, int64, float32, float64）
  - 向后兼容性测试（confusion_matrix_v1）
  - 错误消息格式验证

- **已知风险/缺失信息**：
  - 浮点数标签的自动转换行为未明确
  - 权重参数的数据类型转换细节不明确
  - 稀疏张量输入支持情况未说明
  - 内存使用和性能注意事项缺失