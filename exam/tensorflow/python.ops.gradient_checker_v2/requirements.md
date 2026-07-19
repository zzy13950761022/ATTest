# tensorflow.python.ops.gradient_checker_v2 测试需求

## 1. 目标与范围
- 主要功能与期望行为：数值验证函数梯度计算正确性，比较理论梯度（自动微分）与数值梯度（有限差分）
- 不在范围内的内容：高阶梯度的验证、自定义梯度函数的验证、分布式环境下的梯度检查

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - compute_gradient: f(callable), x(list/tuple), delta(float/None=1/1024)
  - max_error: grad1(list), grad2(list)
- 有效取值范围/维度/设备要求：
  - 数据类型：float16, bfloat16, float32, float64, complex64, complex128
  - 支持TensorFlow 1.x和2.x执行模式
  - 支持IndexedSlices稀疏梯度
- 必需与可选组合：
  - compute_gradient: f和x必需，delta可选
  - max_error: grad1和grad2必需
- 随机性/全局状态要求：无全局状态依赖，数值梯度计算具有确定性

## 3. 输出与判定
- 期望返回结构及关键字段：
  - compute_gradient: 元组(theoretical_gradients, numerical_gradients)，每个为2D numpy数组列表
  - max_error: float类型最大元素差距
- 容差/误差界（如浮点）：
  - 理论梯度与数值梯度差异应在delta量级内
  - 复数类型视为两倍长度实数向量处理
- 状态变化或副作用检查点：
  - 日志记录（vlog级别1）需要验证
  - 无持久化状态改变

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非callable的f参数
  - 无法转换为Tensor的x值
  - grad1和grad2形状不匹配
  - 不支持的数据类型
- 边界值（空、None、0长度、极端形状/数值）：
  - 空张量或零维张量
  - delta为0或负值
  - 极大/极小数值的稳定性
  - 复数输入的边界情况

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：无外部依赖
- 需要mock/monkeypatch的部分：
  - `tensorflow.python.ops.gradient_checker_v2._compute_theoretical_jacobian`
  - `tensorflow.python.ops.gradient_checker_v2._compute_numeric_jacobian`
  - `tensorflow.python.ops.gradient_checker_v2._eval_indexed_slices`
  - `tensorflow.python.ops.gradient_checker_v2._to_numpy`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 基本标量函数梯度验证
  2. 向量/矩阵函数梯度验证
  3. 复数类型梯度计算
  4. IndexedSlices稀疏梯度处理
  5. 不同delta值对数值梯度的影响
- 可选路径（中/低优先级合并为一组列表）：
  - 空张量梯度检查
  - 混合精度类型组合
  - 极端数值稳定性测试
  - 嵌套函数梯度验证
  - 自定义梯度函数兼容性
- 已知风险/缺失信息（仅列条目，不展开）：
  - 高阶梯度的支持情况
  - 分布式环境梯度检查
  - 内存使用峰值监控
  - 大张量计算性能