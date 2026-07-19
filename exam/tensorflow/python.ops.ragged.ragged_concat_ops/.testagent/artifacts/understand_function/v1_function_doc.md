# tensorflow.python.ops.ragged.ragged_concat_ops - 函数说明

## 1. 基本信息
- **FQN**: tensorflow.python.ops.ragged.ragged_concat_ops
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/tensorflow/python/ops/ragged/ragged_concat_ops.py`
- **签名**: 模块包含两个主要函数：
  - `concat(values: typing.List[ragged_tensor.RaggedOrDense], axis, name=None)`
  - `stack(values: typing.List[ragged_tensor.RaggedOrDense], axis=0, name=None)`
- **对象类型**: Python 模块

## 2. 功能概述
- `concat`: 沿指定维度连接不规则张量列表，保持输入张量的秩不变
- `stack`: 沿指定维度堆叠不规则张量列表，输出张量秩增加1
- 两者都支持混合使用规则张量和不规则张量

## 3. 参数说明
**concat 函数参数:**
- `values` (List[RaggedOrDense]): 不规则张量列表，不能为空
- `axis` (int): 连接维度，必须静态已知
- `name` (str/None): 可选操作名称

**stack 函数参数:**
- `values` (List[RaggedOrDense]): 张量列表，不能为空
- `axis` (int, 默认0): 堆叠维度，必须静态已知
- `name` (str/None): 可选操作名称

## 4. 返回值
- 返回 `RaggedTensor` 对象
- `concat`: 输出秩与输入相同
- `stack`: 输出秩为输入秩+1（R>0时）
- 当 R==0 时，`stack` 返回 1D Tensor

## 5. 文档要点
- 所有输入必须具有相同秩和 dtype
- 支持任意形状（与标准 tf.concat/tf.stack 不同）
- axis 参数必须静态已知
- 负 axis 值仅在至少一个输入秩静态已知时支持
- 输入列表不能为空

## 6. 源码摘要
- 主要辅助函数：`_ragged_stack_concat_helper`
- 关键分支：axis=0, axis=1, axis>1 三种情况处理
- 依赖：`ragged_tensor`, `array_ops`, `check_ops`, `math_ops`
- 特殊处理：单输入、全规则张量、秩为1等情况
- 无 I/O、随机性或全局状态副作用

## 7. 示例与用法
**concat 示例:**
```python
t1 = tf.ragged.constant([[1, 2], [3, 4, 5]])
t2 = tf.ragged.constant([[6], [7, 8, 9]])
tf.concat([t1, t2], axis=0)  # [[1, 2], [3, 4, 5], [6], [7, 8, 9]]
tf.concat([t1, t2], axis=1)  # [[1, 2, 6], [3, 4, 5, 7, 8, 9]]
```

**stack 示例:**
```python
tf.ragged.stack([t1, t2], axis=0)  # [[[1, 2], [3, 4, 5]], [[6], [7, 8, 9]]]
tf.ragged.stack([t1, t2], axis=1)  # [[[1, 2], [6]], [[3, 4, 5], [7, 8, 9]]]
```

## 8. 风险与空白
- `RaggedOrDense` 类型定义未在源码中明确展示
- 缺少对 `row_splits_dtype` 匹配过程的详细说明
- 需要测试的边界情况：
  - 空输入列表（应抛出 ValueError）
  - axis 超出范围的情况
  - 混合 dtype 输入
  - 不同秩的输入张量
  - 负 axis 值的处理
  - 秩为0和1的特殊情况
- 模块包含多个函数实体，测试需覆盖 `concat` 和 `stack` 两个主要 API