# tensorflow.python.ops.random_ops - 函数说明

## 1. 基本信息
- **FQN**: tensorflow.python.ops.random_ops
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/tensorflow/python/ops/random_ops.py`
- **签名**: 模块（包含多个随机数生成函数）
- **对象类型**: module

## 2. 功能概述
TensorFlow 随机数生成操作模块，提供多种概率分布的随机数生成功能。包括正态分布、均匀分布、伽马分布、泊松分布等。支持可重复的随机数生成和种子控制。

## 3. 参数说明
模块包含多个函数，核心函数参数：
- **random_normal**: shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, seed=None, name=None
- **random_uniform**: shape, minval=0, maxval=None, dtype=dtypes.float32, seed=None, name=None
- **truncated_normal**: shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, seed=None, name=None
- **random_gamma**: shape, alpha, beta=None, dtype=dtypes.float32, seed=None, name=None
- **random_poisson_v2**: shape, lam, dtype=dtypes.float32, seed=None, name=None

## 4. 返回值
各函数返回指定形状和数据类型的张量：
- 正态分布：符合 N(mean, stddev²) 分布的随机数
- 均匀分布：在 [minval, maxval) 区间内的均匀分布随机数
- 截断正态分布：丢弃超过均值2个标准差外的样本
- 伽马分布：Gamma(alpha, beta) 分布的随机数
- 泊松分布：Poisson(lam) 分布的随机数

## 5. 文档要点
- shape 必须是1-D整数张量或Python数组
- dtype 支持 float16, bfloat16, float32, float64（浮点类型）
- 整数类型仅支持 int32, int64（仅 random_uniform）
- seed 参数控制随机数生成的可重复性
- 广播规则：mean/stddev, minval/maxval 必须可广播到输出形状

## 6. 源码摘要
- 核心函数通过 gen_random_ops 调用底层C++实现
- 使用 random_seed.get_seed(seed) 获取种子对
- 浮点运算：标准正态分布 → 缩放 → 平移
- 形状处理：tensor_util.shape_tensor 转换形状
- 静态形状设置：tensor_util.maybe_set_static_shape

## 7. 示例与用法
```python
# 正态分布
tf.random.normal([4], 0, 1, tf.float32)

# 均匀分布  
tf.random.uniform([2], minval=-1., maxval=0.)

# 截断正态分布
tf.random.truncated_normal([2], mean=3, stddev=1)

# 伽马分布
tf.random.gamma([10], [0.5, 1.5])

# 泊松分布
tf.random.poisson([10], [0.5, 1.5])
```

## 8. 风险与空白
- 目标为模块而非单个函数，包含多个随机数生成函数
- 需要测试多个核心函数：random_normal, random_uniform, truncated_normal, random_gamma, random_poisson_v2
- 未提供所有函数的完整类型注解
- 随机性测试需要特殊处理（统计检验而非精确值匹配）
- 种子机制复杂性：全局种子 vs 操作级种子
- 浮点精度问题：不同硬件/后端可能产生微小差异
- 边界情况：极小/极大参数值、形状为0的情况
- 广播规则的完整测试覆盖
- 整数均匀分布的偏差说明（当 maxval-minval 不是2的幂时）
- 伽马分布中 alpha << 1 或 beta >> 1 时的数值稳定性问题