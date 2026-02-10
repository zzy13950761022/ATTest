# tensorflow.python.util.nest - 函数说明

## 1. 基本信息
- **FQN**: tensorflow.python.util.nest
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/tensorflow/python/util/nest.py`
- **签名**: 模块（包含多个函数）
- **对象类型**: module

## 2. 功能概述
`tf.nest` 模块提供处理嵌套结构（nested structures）的函数。嵌套结构可以是 Python 集合（列表、元组、字典等）或原子值（Tensor、int、float 等）。模块支持结构扁平化、结构映射、结构比较等操作。

## 3. 参数说明
模块包含多个函数，主要参数模式：
- `structure`: 嵌套结构或原子值
- `shallow_tree`: 浅层结构用于部分操作
- `func`: 应用于结构元素的函数
- `check_types`: 布尔值，是否检查类型一致性（默认 True）
- `expand_composites`: 布尔值，是否展开复合张量（默认 False）

## 4. 返回值
各函数返回类型不同：
- `flatten()`: 返回扁平化列表
- `map_structure()`: 返回应用函数后的新结构
- `assert_same_structure()`: 无返回值，失败时抛出异常
- `pack_sequence_as()`: 返回按指定结构打包的序列

## 5. 文档要点
- 嵌套结构定义：Sequence（除字符串/字节）、Mapping（可排序键）、MappingView、attr.s 类
- 原子类型：set、dataclass、tf.Tensor、numpy.array 等
- 结构必须形成树，不能有循环引用
- 字典按键排序以确保确定性行为
- 复合张量（SparseTensor、RaggedTensor）可通过 expand_composites 展开

## 6. 源码摘要
- 核心依赖：`_pywrap_utils` 和 `_pywrap_nest` C++ 扩展
- 关键辅助函数：`_sequence_like`、`_yield_value`、`_yield_sorted_items`
- 递归遍历结构，处理不同类型（Mapping、Sequence、namedtuple、attrs）
- 无 I/O 操作，无随机性，无全局状态修改

## 7. 示例与用法
```python
# 扁平化嵌套结构
tf.nest.flatten([(1, 2), [3, 4]])  # [1, 2, 3, 4]

# 结构映射
tf.nest.map_structure(lambda x: x*2, {'a': 1, 'b': 2})  # {'a': 2, 'b': 4}

# 结构比较
tf.nest.assert_same_structure([1, 2], ['a', 'b'])  # 通过

# 打包序列
tf.nest.pack_sequence_as([0, 0], [1, 2])  # [1, 2]
```

## 8. 风险与空白
- 目标为模块而非单个函数，包含 20+ 个公共函数
- 需要测试多个核心函数：`flatten`、`map_structure`、`assert_same_structure`、`pack_sequence_as`、`is_nested`
- 类型注解信息不完整，依赖运行时检查
- 未明确处理的边界情况：超大嵌套深度、特殊 Python 对象
- 复合张量展开行为需要 TensorFlow 特定测试
- 循环引用行为未定义，测试需避免
- 字典非排序键的处理未完全明确