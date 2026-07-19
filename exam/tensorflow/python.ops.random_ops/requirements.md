# tensorflow.python.ops.random_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试随机数生成模块的核心函数（random_normal, random_uniform, truncated_normal, random_gamma, random_poisson_v2）的正确性、随机性统计特性、种子控制机制
- 不在范围内的内容：底层C++实现细节、非核心辅助函数、其他概率分布函数

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - shape: 1-D整数张量或Python数组，必需
  - mean/stddev: 标量或可广播张量，默认0.0/1.0
  - minval/maxval: 标量或可广播张量，默认0/None
  - alpha/beta: 标量或可广播张量，beta默认None
  - lam: 标量或可广播张量
  - dtype: float16/bfloat16/float32/float64，默认float32
  - seed: 整数或整数对，默认None
  - name: 字符串，默认None

- 有效取值范围/维度/设备要求：
  - shape元素必须为非负整数
  - stddev > 0, minval < maxval, alpha > 0, beta > 0, lam > 0
  - 浮点dtype仅支持浮点类型，整数dtype仅支持int32/int64（仅random_uniform）
  - 支持CPU/GPU设备

- 必需与可选组合：
  - shape必需，其他参数有默认值
  - random_uniform中maxval为None时使用dtype范围

- 随机性/全局状态要求：
  - seed=None使用全局随机状态
  - seed=整数使用确定性随机数生成
  - 种子对(seed1, seed2)用于完整控制

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 返回指定shape和dtype的Tensor
  - 正态分布：符合N(mean, stddev²)分布
  - 均匀分布：在[minval, maxval)区间均匀分布
  - 截断正态：丢弃超过均值2个标准差的样本
  - 伽马分布：符合Gamma(alpha, beta)分布
  - 泊松分布：符合Poisson(lam)分布

- 容差/误差界（如浮点）：
  - 浮点比较使用相对容差1e-6
  - 统计检验使用显著性水平0.05
  - 不同硬件允许微小数值差异

- 状态变化或副作用检查点：
  - 种子机制不改变全局随机状态
  - 相同种子产生相同随机序列
  - 不同种子产生不同随机序列

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - shape包含负数或非整数
  - stddev <= 0, minval >= maxval
  - alpha <= 0, beta <= 0, lam <= 0
  - 不支持的dtype类型
  - 广播失败的不兼容形状

- 边界值（空、None、0长度、极端形状/数值）：
  - shape为[]或[0]（空张量）
  - 极大shape（内存限制）
  - 极小/极大参数值（接近0或极大值）
  - 浮点下溢/上溢情况

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow运行时环境
  - 可选的GPU设备支持
  - 无网络/文件依赖

- 需要mock/monkeypatch的部分：
  - `tensorflow.python.framework.random_seed.get_seed`
  - `tensorflow.python.ops.gen_random_ops`（底层C++操作）
  - `tensorflow.python.framework.tensor_util.shape_tensor`
  - `tensorflow.python.framework.tensor_util.maybe_set_static_shape`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 基本形状和dtype的正确性验证
  2. 种子机制的可重复性测试
  3. 统计分布特性的假设检验
  4. 参数广播规则的正确实现
  5. 边界值和异常输入的错误处理

- 可选路径（中/低优先级合并为一组列表）：
  - 不同硬件（CPU/GPU）的一致性
  - 极端参数值的数值稳定性
  - 大形状张量的性能基准
  - 随机数序列的独立性检验
  - 与其他随机数库的交叉验证

- 已知风险/缺失信息（仅列条目，不展开）：
  - 整数均匀分布的偏差问题
  - 伽马分布alpha<<1的数值稳定性
  - 浮点精度在不同后端的差异
  - 全局种子与操作种子的交互
  - 截断正态的拒绝采样效率