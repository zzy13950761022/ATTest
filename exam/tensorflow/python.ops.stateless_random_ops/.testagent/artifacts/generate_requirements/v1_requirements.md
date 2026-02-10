# tensorflow.python.ops.stateless_random_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试确定性无状态随机数生成操作，验证相同种子和形状下产生相同输出，覆盖均匀分布、正态分布、截断正态分布、分类分布等函数
- 不在范围内的内容：有状态随机操作、非确定性随机数生成器、第三方随机库

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - shape: 1-D integer Tensor/array，必需
  - seed: shape [2] Tensor，dtype int32/int64，必需
  - minval: dtype 类型，浮点数默认 0，整数类型需显式指定
  - maxval: dtype 类型，浮点数默认 1，整数类型需显式指定
  - dtype: tf.dtype，支持 float16/32/64, int32/64, uint32/64
  - alg: str/int/Algorithm，可选 "philox", "threefry", "auto_select"
- 有效取值范围/维度/设备要求：
  - 种子必须是形状为 [2] 的张量
  - XLA 环境下只允许 int32 类型种子
  - 整数类型时，minval 和 maxval 必须同时指定或同时为 None
  - 无符号整数类型 (uint32/64) 不能与 minval/maxval 一起使用
- 必需与可选组合：
  - shape 和 seed 为必需参数
  - 浮点数类型：minval/maxval 可选，默认 [0, 1)
  - 整数类型：minval/maxval 必须同时指定或同时为 None
- 随机性/全局状态要求：无全局状态，完全确定性

## 3. 输出与判定
- 期望返回结构及关键字段：指定形状的张量，填充均匀分布的随机值
- 容差/误差界（如浮点）：
  - 浮点数范围 [minval, maxval)
  - 整数范围 [minval, maxval)
  - 对于全范围整数，minval 和 maxval 需同时为 None
- 状态变化或副作用检查点：无 I/O 操作，无全局状态变化

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 种子形状不为 [2]
  - 整数类型时 minval/maxval 不一致（一个指定一个为 None）
  - 无符号整数类型与 minval/maxval 同时使用
  - XLA 环境下使用 int64 类型种子
- 边界值（空、None、0 长度、极端形状/数值）：
  - shape 为空列表或包含 0 的维度
  - minval = maxval 的情况
  - 极大/极小数值范围
  - 种子值为边界值（0, -1, 最大/最小整数值）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：无外部依赖
- 需要 mock/monkeypatch 的部分：
  - `tensorflow.python.ops.gen_stateless_random_ops.stateless_random_uniform_v2`
  - `tensorflow.python.ops.gen_stateless_random_ops.stateless_random_uniform_int_v2`
  - `tensorflow.python.ops.gen_stateless_random_ops.stateless_random_uniform_full_int_v2`
  - `tensorflow.python.ops.gen_stateless_random_ops.stateless_random_normal_v2`
  - `tensorflow.python.ops.gen_stateless_random_ops.stateless_truncated_normal_v2`
  - `tensorflow.python.ops.gen_stateless_random_ops.stateless_multinomial_v2`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 相同种子和形状产生相同输出（确定性验证）
  2. 浮点数均匀分布范围验证 [minval, maxval)
  3. 整数类型 minval/maxval 一致性约束
  4. 种子形状 [2] 的强制要求
  5. 无符号整数类型与 minval/maxval 互斥性
- 可选路径（中/低优先级合并为一组列表）：
  - 不同算法（philox, threefry, auto_select）的行为差异
  - XLA 编译环境下的约束验证
  - 极端形状和数值范围的边界测试
  - 广播行为的完整测试覆盖
  - 整数类型的偏差问题（当 maxval-minval 不是 2 的幂时）
  - 跨设备一致性测试（CPU, GPU, TPU）
  - 性能基准测试（大规模形状）
- 已知风险/缺失信息（仅列条目，不展开）：
  - 算法参数 "auto_select" 的行为可能因设备而异
  - 整数类型的偏差问题文档说明不充分
  - 无符号整数类型的使用限制未在函数签名中明确标注
  - XLA 兼容性约束仅在文档中提及