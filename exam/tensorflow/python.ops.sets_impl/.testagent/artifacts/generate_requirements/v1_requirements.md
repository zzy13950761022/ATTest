# tensorflow.python.ops.sets_impl 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证稀疏张量集合操作（交集、并集、差集、大小计算）的正确性，支持稀疏/密集张量混合操作
- 不在范围内的内容：梯度计算（NotDifferentiable）、非集合张量运算、自定义数据类型扩展

## 2. 输入与约束
- 参数列表：
  - a/b: Tensor/SparseTensor，除最后一维外维度必须匹配
  - validate_indices: bool/True，验证稀疏索引顺序和范围
  - aminusb: bool/True，控制差集方向（仅set_difference）
- 有效取值范围/维度/设备要求：
  - 数据类型：int8, int16, int32, int64, uint8, uint16, string
  - 稀疏索引必须按行主序排序
  - 支持 DenseTensor,SparseTensor 顺序，不支持 SparseTensor,DenseTensor 顺序
- 必需与可选组合：a,b 必需，validate_indices 可选（默认True）
- 随机性/全局状态要求：无随机性，无全局状态依赖

## 3. 输出与判定
- 期望返回结构：
  - set_size: int32 Tensor，形状为输入张量秩减1
  - set_intersection/difference/union: SparseTensor，与输入张量秩相同
- 容差/误差界：整数/字符串精确匹配，无浮点误差
- 状态变化或副作用检查点：无副作用，纯函数操作

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常：
  - 数据类型不在 _VALID_DTYPES 中
  - 稀疏索引未按行主序排序（validate_indices=True时）
  - 输入维度不匹配（除最后一维外）
  - 不支持 SparseTensor,DenseTensor 顺序
- 边界值：
  - 空集合（零元素稀疏张量）
  - 单元素集合
  - 极端形状（高维稀疏张量）
  - 大数值（int64边界）
  - 长字符串元素

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：无
- 需要 mock/monkeypatch 的部分：
  - `tensorflow.python.ops.gen_set_ops.set_size`
  - `tensorflow.python.ops.gen_set_ops.sparse_to_sparse_set_operation`
  - `tensorflow.python.framework.sparse_tensor.SparseTensor`
  - `tensorflow.python.ops.sets_impl._convert_to_tensors_or_sparse_tensors`

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. 稀疏-稀疏张量基本集合操作
  2. 密集-稀疏张量混合操作
  3. validate_indices=True/False 的行为差异
  4. set_difference 的 aminusb 参数切换
  5. 空集合和单元素集合边界情况
- 可选路径（中/低优先级）：
  - 所有支持数据类型的遍历测试
  - 高维稀疏张量操作
  - 大尺寸集合性能测试
  - 字符串类型集合操作
  - 索引验证失败的具体错误信息
- 已知风险/缺失信息：
  - 不支持 SparseTensor,DenseTensor 顺序的具体原因
  - 稀疏索引验证的具体算法细节
  - 内存使用和性能约束未定义
  - 底层 C++ 实现细节未暴露