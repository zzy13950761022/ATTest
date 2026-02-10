# tensorflow.python.ops.array_ops 测试需求

## 1. 目标与范围
- **主要功能与期望行为**: 测试 TensorFlow 数组操作模块的核心张量操作功能，包括张量重塑、维度扩展、连接、堆叠等基础操作。验证函数在合法输入下返回正确形状和数值的张量，保持元素总数不变，遵循 TensorFlow 张量计算语义。
- **不在范围内的内容**: 不测试模块中所有 100+ 个函数，仅覆盖核心公共 API；不测试底层 C++ 实现细节；不测试 GPU/TPU 特定硬件加速行为；不测试分布式计算场景。

## 2. 输入与约束
- **参数列表（名称、类型/shape、默认值）**:
  - `reshape(tensor, shape, name=None)`: tensor(Tensor), shape(list/tuple/Tensor), name(str)
  - `expand_dims(input, axis=None, name=None, dim=None)`: input(Tensor), axis(int), name(str), dim(int, deprecated)
  - `concat(values, axis, name="concat")`: values(list of Tensor), axis(int), name(str)
  - `stack(values, axis=0, name="stack")`: values(list of Tensor), axis(int), name(str)
  - `unstack(value, num=None, axis=0, name="unstack")`: value(Tensor), num(int), axis(int), name(str)

- **有效取值范围/维度/设备要求**:
  - 张量元素类型支持数值类型（int, float, complex）
  - reshape 形状参数最多一个维度可为 -1（自动推断）
  - expand_dims 的 axis 必须在 `[-rank(input)-1, rank(input)]` 范围内
  - concat 和 stack 的输入张量列表必须非空且形状兼容
  - 所有操作支持 CPU 设备，GPU/TPU 为可选

- **必需与可选组合**:
  - reshape: tensor 必需，shape 必需，name 可选
  - expand_dims: input 必需，axis 必需，name 可选，dim 已弃用
  - concat: values 必需，axis 必需，name 可选

- **随机性/全局状态要求**:
  - 无随机性操作
  - 不修改全局状态
  - 纯函数式操作，无副作用

## 3. 输出与判定
- **期望返回结构及关键字段**:
  - 返回 Tensor 对象，类型与输入张量相同
  - reshape: 输出形状等于指定 shape，总元素数不变
  - expand_dims: 输出 rank = input_rank + 1，在指定轴插入维度 1
  - concat: 输出形状在连接轴维度等于各输入张量该维度之和
  - stack: 输出 rank = input_rank + 1，新增维度大小为 len(values)

- **容差/误差界（如浮点）**:
  - 数值精度遵循 TensorFlow 浮点运算标准
  - 元素值应完全相等（无浮点误差累积）
  - 形状和类型必须精确匹配

- **状态变化或副作用检查点**:
  - 无 I/O 操作
  - 不修改输入张量
  - 不改变全局计算图状态
  - 不产生内存泄漏

## 4. 错误与异常场景
- **非法输入/维度/类型触发的异常或警告**:
  - 无效形状参数（如负值且不为 -1）
  - 形状不兼容（reshape 元素总数不匹配）
  - 越界 axis 参数
  - 空张量列表（concat, stack）
  - 不兼容的输入张量形状（concat 非连接轴维度不一致）
  - 类型不匹配（非数值类型张量）

- **边界值（空、None、0 长度、极端形状/数值）**:
  - 空张量（shape=[]）
  - 零维张量（标量）
  - 极大形状参数（接近内存限制）
  - 负索引边界（axis=-rank-1）
  - 形状参数为 None
  - 自动推断维度（shape 包含 -1）

## 5. 依赖与环境
- **外部资源/设备/网络/文件依赖**:
  - TensorFlow 运行时环境
  - 计算设备（CPU 必需，GPU/TPU 可选）
  - 无网络或文件系统依赖

- **需要 mock/monkeypatch 的部分**:
  - `tensorflow.python.ops.gen_array_ops.reshape`
  - `tensorflow.python.ops.gen_array_ops.expand_dims`
  - `tensorflow.python.ops.gen_array_ops.concat_v2`
  - `tensorflow.python.ops.gen_array_ops.pack`
  - `tensorflow.python.ops.gen_array_ops.unpack`
  - `tensorflow.python.framework.tensor_util.maybe_set_static_shape`
  - `tensorflow.python.framework.ops.convert_to_tensor`
  - `tensorflow.python.framework.ops.Tensor`
  - `tensorflow.python.framework.dtypes.as_dtype`
  - `tensorflow.python.framework.constant_op.constant`

## 6. 覆盖与优先级
- **必测路径（高优先级，最多 5 条，短句）**:
  1. reshape 基本形状变换与 -1 自动推断
  2. expand_dims 正负轴索引插入维度
  3. concat 多张量沿指定轴连接
  4. stack 张量列表堆叠为新维度
  5. 异常输入触发正确错误类型和消息

- **可选路径（中/低优先级合并为一组列表）**:
  - 大张量性能基准测试
  - 复杂嵌套形状操作
  - 混合数据类型操作
  - 弃用参数兼容性测试
  - GPU/TPU 设备特定行为
  - 计算图模式与 eager 模式一致性
  - 梯度计算正确性

- **已知风险/缺失信息（仅列条目，不展开）**:
  - 模块无 `__all__` 定义，公共 API 边界模糊
  - 部分函数缺少详细类型注解
  - v1/v2 版本函数差异
  - 底层 C++ 操作实现细节不透明
  - 内存使用峰值未文档化
  - 极端形状下的性能退化点