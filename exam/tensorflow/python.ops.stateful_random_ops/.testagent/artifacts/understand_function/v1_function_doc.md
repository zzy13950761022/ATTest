# tensorflow.python.ops.stateful_random_ops - 函数说明

## 1. 基本信息
- **FQN**: tensorflow.python.ops.stateful_random_ops
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/tensorflow/python/ops/stateful_random_ops.py`
- **签名**: 模块（包含多个类和函数）
- **对象类型**: module

## 2. 功能概述
提供有状态随机数生成操作，核心是 `Generator` 类，用于生成各种分布的随机数。支持 Philox 和 ThreeFry 两种 RNG 算法，可管理内部状态并支持分布式环境。

## 3. 参数说明
模块包含多个函数和类，主要实体：
- `Generator` 类：主要随机数生成器
  - 构造函数参数：`copy_from`, `state`, `alg`
  - 类方法：`from_seed`, `from_non_deterministic_state`, `from_key_counter`
- `non_deterministic_ints` 函数：生成非确定性整数
  - `shape`: 输出张量形状
  - `dtype`: 输出数据类型（默认 int64）
- `create_rng_state` 函数：从种子创建 RNG 状态
  - `seed`: 整数或 1-D numpy 数组
  - `alg`: RNG 算法（字符串、Algorithm 或整数）

## 4. 返回值
- `Generator` 实例：提供各种随机分布方法
- `non_deterministic_ints`: 指定形状的随机整数张量
- `create_rng_state`: 1-D 状态张量，大小取决于算法

## 5. 文档要点
- 支持两种算法：Philox (RNG_ALG_PHILOX=1) 和 ThreeFry (RNG_ALG_THREEFRY=2)
- 种子为 1024 位无符号整数，短种子会填充
- 状态变量使用 int64 类型，避免 GPU 限制
- 在分布式策略中，每个副本获得不同的随机数流
- 全局生成器可通过 `get_global_generator` 访问

## 6. 源码摘要
- 核心类 `Generator` 使用 `tf.Variable` 管理状态
- 通过 `_prepare_key_counter` 准备密钥和计数器
- 依赖底层 C++ 操作：`gen_stateful_random_ops`, `gen_stateless_random_ops_v2`
- 副作用：每次生成随机数都会更新内部状态
- 支持分布式环境下的状态同步

## 7. 示例与用法
```python
# 从种子创建生成器
g = tf.random.Generator.from_seed(1234)
g.normal(shape=(2, 3))

# 非确定性状态
g = tf.random.Generator.from_non_deterministic_state()

# 全局生成器
g = tf.random.get_global_generator()
```

## 8. 风险与空白
- 模块包含多个实体（类、函数、常量），测试需覆盖主要 API
- 算法选择逻辑未完全明确（TODO 注释）
- 某些边界条件文档不详细（如种子范围限制）
- 分布式策略下的行为复杂，需要多场景测试
- 缺少对不支持算法的错误处理细节
- 状态形状验证依赖于内部函数 `_get_state_size`
- 整数随机数生成存在轻微偏差（除非范围是 2 的幂）
- 浮点数范围可能包含上边界（由于舍入）
