# tensorflow.python.ops.stateful_random_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 测试 Generator 类及其方法：构造函数、from_seed、from_non_deterministic_state、from_key_counter
  - 验证随机数生成器状态管理（tf.Variable 存储）
  - 测试两种 RNG 算法：Philox (RNG_ALG_PHILOX=1) 和 ThreeFry (RNG_ALG_THREEFRY=2)
  - 验证分布式环境下每个副本获得不同随机数流
  - 测试全局生成器访问（get_global_generator）
  - 验证 non_deterministic_ints 函数生成非确定性整数
  - 测试 create_rng_state 函数从种子创建状态张量

- 不在范围内的内容
  - 底层 C++ 操作（gen_stateful_random_ops, gen_stateless_random_ops_v2）的内部实现
  - 第三方 RNG 算法扩展
  - 非 TensorFlow 环境的兼容性
  - 性能基准测试（仅功能正确性）

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - Generator 构造函数：copy_from（Generator）、state（1-D 张量）、alg（字符串/Algorithm/整数）
  - from_seed：seed（整数或 1-D numpy 数组）、alg（默认 RNG_ALG_PHILOX）
  - from_key_counter：key（2 元素数组）、counter（3 元素数组）、alg
  - non_deterministic_ints：shape（张量形状）、dtype（默认 int64）
  - create_rng_state：seed（整数或 1-D 数组）、alg

- 有效取值范围/维度/设备要求
  - 种子：1024 位无符号整数，短种子自动填充
  - 状态变量：int64 类型（避免 GPU 限制）
  - key：2 元素数组，counter：3 元素数组
  - 算法标识：整数 1（Philox）或 2（ThreeFry）
  - 支持 CPU 和 GPU 设备

- 必需与可选组合
  - Generator 构造函数：state 和 alg 必须同时提供或同时省略
  - from_seed：seed 必需，alg 可选
  - from_key_counter：key、counter、alg 全部必需

- 随机性/全局状态要求
  - 每次生成随机数都会更新内部状态
  - 全局生成器状态在会话间保持
  - 分布式策略中状态同步机制

## 3. 输出与判定
- 期望返回结构及关键字段
  - Generator 实例：包含 state 属性和各种分布方法
  - non_deterministic_ints：指定形状的随机整数张量
  - create_rng_state：1-D 状态张量，大小由算法决定
  - 随机分布方法：normal、uniform、poisson 等返回指定形状张量

- 容差/误差界（如浮点）
  - 浮点数范围可能包含上边界（由于舍入）
  - 整数随机数存在轻微偏差（除非范围是 2 的幂）
  - 分布参数验证：均值、方差在统计容差内

- 状态变化或副作用检查点
  - 每次 random 调用后 state 属性必须更新
  - 相同种子生成相同随机序列
  - 不同种子生成不同随机序列
  - 状态复制（copy_from）产生独立生成器

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 无效算法标识（非 1 或 2）
  - 状态张量形状不正确（非 1-D）
  - key/counter 数组维度错误
  - 不支持的 dtype 参数
  - 无效的 shape 参数（负值、非整数）

- 边界值（空、None、0 长度、极端形状/数值）
  - 空种子数组
  - 极大/极小种子值
  - shape 为 0 或包含 0 的维度
  - 极端分布参数（如方差为 0）
  - 状态张量为 None 或空张量

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 运行时环境
  - GPU 设备（可选，用于 GPU 测试）
  - 分布式策略环境（如 tf.distribute）

- 需要 mock/monkeypatch 的部分
  - `tensorflow.python.ops.gen_stateful_random_ops`（底层 C++ 操作）
  - `tensorflow.python.ops.gen_stateless_random_ops_v2`
  - `tensorflow.Variable`（状态存储）
  - `tensorflow.distribute.get_strategy`（分布式环境）
  - `tensorflow.config.experimental.get_synchronous_execution`（执行模式）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. Generator.from_seed 创建生成器并生成随机序列
  2. 相同种子产生相同序列，不同种子产生不同序列
  3. 状态更新验证：每次 random 调用后 state 变化
  4. 两种算法（Philox/ThreeFry）基本功能
  5. 错误输入触发适当异常（无效算法、错误形状）

- 可选路径（中/低优先级合并为一组列表）
  - 分布式策略下的状态行为
  - 全局生成器访问和重置
  - 极端形状和参数值
  - 不同设备（CPU/GPU）一致性
  - 状态复制和独立操作
  - 非确定性状态生成
  - key_counter 方式创建生成器
  - 各种分布方法（normal、uniform、poisson 等）
  - 种子自动填充机制
  - 状态张量形状验证

- 已知风险/缺失信息（仅列条目，不展开）
  - 算法选择逻辑文档不完整（TODO 注释）
  - 种子范围限制未详细说明
  - 不支持算法的错误处理细节缺失
  - 状态形状验证依赖内部函数 _get_state_size
  - 分布式策略下的复杂行为需要多场景验证