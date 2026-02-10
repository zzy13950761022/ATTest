# tensorflow.python.data.experimental.ops.resampling 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证 rejection_resample 函数返回数据集转换函数
  - 测试重采样转换实现目标分布调整
  - 验证弃用警告正确触发
  - 确保 class_func 正确映射到类别索引
  - 验证随机种子控制采样可重复性

- 不在范围内的内容
  - 不测试底层 rejection_resample 算法实现细节
  - 不验证 tf.data.Dataset.rejection_resample 替代方法
  - 不进行大规模性能基准测试

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - class_func: 函数，输入数据集元素→tf.int32 标量，范围 [0, num_classes)
  - target_dist: tf.Tensor，浮点类型，形状 [num_classes]
  - initial_dist: tf.Tensor/None，浮点类型，形状 [num_classes]，默认 None
  - seed: int/None，Python 整数，默认 None

- 有效取值范围/维度/设备要求
  - class_func 返回值必须在 [0, num_classes) 范围内
  - target_dist 和 initial_dist 必须为浮点张量
  - num_classes 必须为正整数
  - 分布张量形状必须匹配且为一维
  - 支持 CPU/GPU 设备

- 必需与可选组合
  - class_func 和 target_dist 为必需参数
  - initial_dist 可选，未提供时实时估计分布
  - seed 可选，控制随机性

- 随机性/全局状态要求
  - seed 参数控制采样随机性
  - 相同 seed 应产生相同采样结果
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回函数对象（_apply_fn）
  - 返回函数接受 Dataset 参数，返回转换后 Dataset
  - 转换后数据集元素类型与原始一致

- 容差/误差界（如浮点）
  - 浮点比较容差：1e-6
  - 分布归一化容差：1e-5
  - 采样比例误差：±5%

- 状态变化或副作用检查点
  - 验证弃用警告触发
  - 检查数据集元素数量变化（部分被丢弃）
  - 验证 class_func 调用次数与输入元素匹配

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - class_func 返回越界值（<0 或 ≥num_classes）
  - target_dist 非浮点类型
  - 分布张量形状不匹配
  - 分布张量非一维
  - class_func 返回非标量张量

- 边界值（空、None、0 长度、极端形状/数值）
  - 空数据集输入
  - target_dist 全零或负值
  - num_classes=1 边界情况
  - 极端分布值（接近0或1）
  - seed 为负整数或大整数

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 数据集 API
  - 随机数生成器（依赖 seed）
  - 无外部网络/文件依赖

- 需要 mock/monkeypatch 的部分
  - tf.data.Dataset.rejection_resample 方法
  - 弃用警告检查
  - 随机数生成器（用于确定性测试）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 基本功能：返回转换函数并应用数据集
  2. 弃用警告：验证正确触发弃用提示
  3. 分布调整：验证采样后分布接近 target_dist
  4. 种子控制：相同 seed 产生相同采样结果
  5. 可选参数：initial_dist=None 时实时估计分布

- 可选路径（中/低优先级合并为一组列表）
  - 不同数据类型组合测试
  - 大规模数据集性能验证
  - 多设备（CPU/GPU）兼容性
  - 嵌套数据集结构支持
  - 与 tf.data 其他转换组合使用

- 已知风险/缺失信息（仅列条目，不展开）
  - 文档中缺少具体使用示例
  - 未明确分布归一化要求
  - 缺少异常类型定义
  - 性能影响未量化
  - 内存使用模式未说明