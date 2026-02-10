# tensorflow.python.ops.ragged.segment_id_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - `row_splits_to_segment_ids`: 将行分割数组转换为段ID数组，实现RaggedTensor行分割到段ID的转换
  - `segment_ids_to_row_splits`: 将段ID数组转换为行分割数组，实现段ID到RaggedTensor行分割的转换
  - 确保两个函数互为逆操作：segment_ids_to_row_splits(row_splits_to_segment_ids(x)) ≈ x
- 不在范围内的内容
  - RaggedTensor的其他操作（如拼接、切片、数学运算）
  - 非整数类型的输入处理
  - GPU/TPU特定优化

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `row_splits_to_segment_ids`:
    - splits: 1-D Tensor[int32/int64], 必需
    - name: str/None, 可选, 默认None
    - out_type: dtype/None, 可选, 默认splits.dtype或int64
  - `segment_ids_to_row_splits`:
    - segment_ids: 1-D Tensor[int32/int64], 必需
    - num_segments: scalar int/None, 可选, 默认max(segment_ids)+1
    - out_type: dtype/None, 可选, 默认segment_ids.dtype或int64
    - name: str/None, 可选, 默认None

- 有效取值范围/维度/设备要求
  - splits必须是1-D整数张量，splits[0]必须为0
  - splits必须已排序（非递减）
  - segment_ids必须是1-D整数张量
  - 支持int32和int64数据类型
  - 支持CPU设备，GPU依赖TensorFlow运行时

- 必需与可选组合
  - splits/segment_ids为必需参数
  - name和out_type为可选参数
  - num_segments仅用于segment_ids_to_row_splits，可选

- 随机性/全局状态要求
  - 无随机性
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - `row_splits_to_segment_ids`: 返回1-D整数张量，shape=[splits[-1]]
  - `segment_ids_to_row_splits`: 返回1-D整数张量，shape=[num_segments + 1]
  - 输出类型与输入类型一致或按out_type指定

- 容差/误差界（如浮点）
  - 整数运算，要求精确相等
  - 无浮点容差

- 状态变化或副作用检查点
  - 无副作用
  - 不修改输入张量
  - 不改变全局状态

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - splits非1-D张量：ValueError
  - splits[0] != 0：InvalidArgumentError
  - splits未排序：InvalidArgumentError
  - 空splits数组：ValueError
  - segment_ids非1-D张量：ValueError
  - 非整数类型输入：TypeError
  - num_segments < max(segment_ids)+1：InvalidArgumentError

- 边界值（空、None、0长度、极端形状/数值）
  - splits = [0]：输出空数组[]
  - segment_ids = []：输出[0]
  - 大整数输入（接近int32/int64边界）
  - 零长度段：splits = [0, 0, 0, 5]
  - 单一段：segment_ids = [0, 0, 0, 0]
  - 不连续段ID：segment_ids = [0, 0, 2, 2, 5, 5]

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow运行时
  - 无网络/文件依赖

- 需要mock/monkeypatch的部分
  - `tensorflow.python.ops.bincount_ops.bincount`（segment_ids_to_row_splits内部使用）
  - `tensorflow.python.ops.ragged.ragged_util.repeat`（row_splits_to_segment_ids内部使用）
  - `tensorflow.python.framework.ops.convert_to_tensor`（类型转换）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. 基本正向转换：标准splits/segment_ids输入输出验证
  2. 逆操作验证：row_splits_to_segment_ids和segment_ids_to_row_splits互为逆操作
  3. 数据类型边界：int32和int64输入输出类型一致性
  4. 空/零长度边界：空数组、零长度段处理
  5. 错误输入验证：非法splits[0]、未排序、非1-D张量

- 可选路径（中/低优先级合并为一组列表）
  - 大整数输入（接近数据类型边界）
  - 不连续段ID处理
  - num_segments显式指定与默认值差异
  - out_type参数覆盖默认类型
  - name参数功能验证
  - 多设备支持（CPU/GPU可用性）

- 已知风险/缺失信息（仅列条目，不展开）
  - segment_ids是否必须连续未明确说明
  - num_segments为None时的边界情况处理
  - 大整数输入时的溢出风险
  - bincount_ops.bincount的minlength/maxlength参数行为
  - int64到int32类型转换的截断细节