# tensorflow.python.ops.manip_ops - 函数说明

## 1. 基本信息
- **FQN**: tensorflow.python.ops.manip_ops
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/tensorflow/python/ops/manip_ops.py`
- **签名**: roll(input, shift, axis, name=None)
- **对象类型**: 模块（包含单个核心函数 `roll`）

## 2. 功能概述
- 沿指定轴滚动张量元素
- 元素按偏移量正向（向大索引）移动，负偏移反向移动
- 超出边界的元素会循环到另一端

## 3. 参数说明
- **input** (Tensor): 任意类型的输入张量
- **shift** (Tensor[int32/int64]): 0-D 或 1-D 张量，指定滚动偏移量
  - `shift[i]` 对应 `axis[i]` 维度的滚动量
  - 负值表示反向滚动
- **axis** (Tensor[int32/int64]): 0-D 或 1-D 张量，指定滚动轴
  - `axis[i]` 指定 `shift[i]` 作用的维度
  - 同一轴多次出现时，偏移量累加
- **name** (str/None): 操作名称（可选）

## 4. 返回值
- **类型**: Tensor，与输入张量类型相同
- **结构**: 与输入张量形状相同，元素位置重新排列

## 5. 文档要点
- 支持多轴同时滚动
- 同一轴可多次指定，偏移量累加
- 元素循环：超出边界元素从另一端出现
- 输入张量类型任意，shift/axis 必须是 int32/int64

## 6. 源码摘要
- 关键路径：直接调用 `_gen_manip_ops.roll()`
- 依赖：`gen_manip_ops` C++ 实现
- 装饰器：`@tf_export`、`@dispatch.add_dispatch_support`、`@deprecation.deprecated_endpoints`
- 副作用：无 I/O、随机性或全局状态修改

## 7. 示例与用法
```python
# 一维滚动
# t = [0, 1, 2, 3, 4]
roll(t, shift=2, axis=0) → [3, 4, 0, 1, 2]

# 多轴滚动
# t = [[0,1,2,3,4], [5,6,7,8,9]]
roll(t, shift=[1,-2], axis=[0,1]) → [[7,8,9,5,6], [2,3,4,0,1]]

# 同轴多次滚动
roll(t, shift=[2,-3], axis=[1,1]) → [[1,2,3,4,0], [6,7,8,9,5]]
```

## 8. 风险与空白
- **多实体情况**: 模块仅导出 `roll` 函数，需测试该核心函数
- **类型约束**: 缺少对输入张量形状的具体约束说明
- **边界情况**: 未明确处理空张量、零维度等情况
- **性能影响**: 未说明大张量或高维度的性能特性
- **错误处理**: 缺少参数验证和异常类型文档
- **设备限制**: 未说明 GPU/TPU 支持情况