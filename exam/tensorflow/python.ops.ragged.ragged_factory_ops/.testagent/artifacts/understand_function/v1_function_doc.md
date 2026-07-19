# tensorflow.python.ops.ragged.ragged_factory_ops - 函数说明

## 1. 基本信息
- **FQN**: tensorflow.python.ops.ragged.ragged_factory_ops
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/tensorflow/python/ops/ragged/ragged_factory_ops.py`
- **签名**: 模块包含多个函数，核心函数为：
  - `constant(pylist, dtype=None, ragged_rank=None, inner_shape=None, name=None, row_splits_dtype=dtypes.int64)`
  - `constant_value(pylist, dtype=None, ragged_rank=None, inner_shape=None, row_splits_dtype="int64")`
  - `placeholder(dtype, ragged_rank, value_shape=None, name=None)`
- **对象类型**: Python 模块

## 2. 功能概述
- `constant`: 从嵌套 Python 列表构造常量 RaggedTensor
- `constant_value`: 从嵌套 Python 列表构造 RaggedTensorValue（NumPy 数组）
- `placeholder`: 创建 RaggedTensor 占位符，用于 TensorFlow 1.x 的 feed_dict

## 3. 参数说明
**constant 函数参数：**
- `pylist` (list/tuple/np.ndarray): 嵌套列表/元组/数组，非列表元素必须是标量
- `dtype` (tf.DType/None): 返回张量的元素类型，默认根据标量值推断
- `ragged_rank` (int/None): 返回张量的 ragged 秩，必须非负且小于 K
- `inner_shape` (tuple/None): 内部值的形状，默认基于内容推断
- `name` (str/None): 操作名称前缀
- `row_splits_dtype` (tf.int32/tf.int64): row_splits 数据类型，默认 int64

**constant_value 函数参数：**
- 参数与 `constant` 类似，但返回 RaggedTensorValue
- `row_splits_dtype` 接受字符串 "int32" 或 "int64"

**placeholder 函数参数：**
- `dtype`: RaggedTensor 的数据类型
- `ragged_rank`: RaggedTensor 的 ragged 秩
- `value_shape`: 单个平坦值的形状
- `name`: 操作名称

## 4. 返回值
- `constant`: 返回 `tf.RaggedTensor`，具有指定 ragged_rank 和 inner_shape
- `constant_value`: 返回 `tf.RaggedTensorValue` 或 `numpy.array`
- `placeholder`: 返回 `tf.RaggedTensor` 占位符，不能直接求值

## 5. 文档要点
- 所有标量值必须具有相同的嵌套深度 K
- 如果 pylist 不包含标量值，K 比空列表的最大深度大 1
- ragged_rank 必须小于 K
- inner_shape 和 ragged_rank 必须与 pylist 兼容
- 标量值必须与 dtype 兼容

## 6. 源码摘要
- 核心函数：`_constant_value` 处理实际构造逻辑
- 辅助函数：`_find_scalar_and_max_depth` 计算嵌套深度
- 辅助函数：`_default_inner_shape_for_pylist` 推断默认 inner_shape
- 依赖：`ragged_tensor.RaggedTensor.from_row_splits` 创建 RaggedTensor
- 依赖：`constant_op.constant` 创建常量张量

## 7. 示例与用法
```python
# constant 示例
>>> tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
<tf.RaggedTensor [[1, 2], [3], [4, 5, 6]]>

# constant_value 示例
>>> tf.compat.v1.ragged.constant_value([[1, 2], [3], [4, 5, 6]])
tf.RaggedTensorValue(values=array([1, 2, 3, 4, 5, 6]),
                     row_splits=array([0, 2, 3, 6]))
```

## 8. 风险与空白
- 模块包含多个函数，测试需覆盖所有三个公共函数
- `constant` 和 `constant_value` 参数逻辑相似但实现不同
- `placeholder` 仅适用于 TensorFlow 1.x 的图模式
- 需要测试边界情况：空列表、不同嵌套深度、无效 ragged_rank
- 需要验证 dtype 推断逻辑和错误处理
- 缺少对复杂嵌套结构（如混合类型）的详细约束说明
- 需要测试 row_splits_dtype 参数的有效性验证