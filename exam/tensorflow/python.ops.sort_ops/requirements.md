# tensorflow.python.ops.sort_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - `sort()`: 对数值张量沿指定轴排序，返回排序后张量
  - `argsort()`: 返回排序索引，用于重建排序结果
  - 支持升序('ASCENDING')和降序('DESCENDING')排序
  - 处理多维张量，可沿任意轴排序
- 不在范围内的内容
  - 非数值类型（字符串、布尔值）排序
  - 动态轴参数（axis必须是常量标量）
  - 稳定排序（stable参数当前未实现）

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - values: Tensor (float/int类型), 必需, 1-D或更高维
  - axis: int, 默认-1（最内层轴）, 必须是常量标量
  - direction: str, 默认'ASCENDING', 仅接受'ASCENDING'或'DESCENDING'
  - stable: bool (仅argsort), 默认False, 当前未实现
  - name: str, 默认None, 操作名称
- 有效取值范围/维度/设备要求
  - values: float16/32/64, int8/16/32/64类型
  - axis: [-rank(values), rank(values)-1]范围内整数
  - 支持CPU/GPU设备
- 必需与可选组合
  - values为必需参数
  - axis, direction, name为可选参数
  - stable仅argsort函数可用
- 随机性/全局状态要求
  - 无随机性要求
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - `sort()`: 与输入相同dtype和shape的张量
  - `argsort()`: int32类型张量，与输入相同shape
- 容差/误差界（如浮点）
  - 浮点类型：允许1e-6相对误差
  - 整数类型：精确匹配
- 状态变化或副作用检查点
  - 无副作用，纯函数
  - 不修改输入张量

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 非数值类型输入引发TypeError
  - 动态axis参数引发ValueError
  - 无效direction值引发ValueError
  - 轴超出范围引发InvalidArgumentError
- 边界值（空、None、0长度、极端形状/数值）
  - 空张量：shape=(0,)或shape=(n,0)
  - 单元素张量
  - 极端数值：inf, -inf, NaN
  - 大整数溢出边界
  - 高维张量（>4维）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow运行时
  - 可选GPU支持
- 需要mock/monkeypatch的部分
  - `tensorflow.python.ops.nn_ops.top_k`（降序排序实现）
  - `tensorflow.python.ops.array_ops.transpose`（轴转置）
  - `tensorflow.python.ops.math_ops.cast`（类型转换）
  - `tensorflow.python.ops.math_ops.negative`（数值变换）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. 1D浮点张量升序/降序排序
  2. 多维张量沿不同轴排序
  3. 整数类型排序及溢出处理
  4. argsort索引正确性验证
  5. 边界轴值（-1, 0, 中间轴）排序
- 可选路径（中/低优先级合并为一组列表）
  - 高维张量（>4维）排序
  - 混合符号整数排序
  - 特殊浮点值（inf, -inf, NaN）排序行为
  - 空张量和单元素张量
  - 大尺寸张量性能测试
- 已知风险/缺失信息（仅列条目，不展开）
  - stable参数未实现但保留
  - NaN排序行为未明确
  - 整数溢出处理细节
  - 非最优化轴排序性能