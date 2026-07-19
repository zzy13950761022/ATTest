# tensorflow.python.ops.sparse_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证稀疏张量与密集张量转换、稀疏矩阵运算、稀疏张量操作的正确性，确保索引排序、值提取、形状转换符合规范
- 不在范围内的内容：底层 C++ 实现、分布式计算、自定义稀疏格式、第三方库集成

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - indices: int32/int64 Tensor [nnz, rank]
  - values: 任意dtype Tensor [nnz]
  - shape: int64 Tensor [rank]
  - dense_tensor: 任意dtype Tensor
  - adjoint_a/adjoint_b: bool (默认False)
  - validate_indices: bool (默认True)
  - name: str (默认None)

- 有效取值范围/维度/设备要求：
  - indices 必须按字典序排序且不重复
  - shape 元素必须为正整数
  - 稀疏矩阵乘法要求矩阵维度兼容
  - 支持 CPU/GPU 设备

- 必需与可选组合：
  - 稀疏张量三要素必须同时提供 (indices, values, shape)
  - 转换函数至少需要一个输入张量
  - 矩阵运算需要两个操作数

- 随机性/全局状态要求：
  - 无随机性要求
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段：
  - SparseTensor: indices, values, shape 属性
  - 密集 Tensor: 与输入形状匹配
  - 矩阵运算结果: 符合数学定义

- 容差/误差界（如浮点）：
  - 浮点运算误差在 1e-6 范围内
  - 整数运算精确匹配
  - 形状转换零误差

- 状态变化或副作用检查点：
  - 无文件系统操作
  - 无网络访问
  - 无全局变量修改

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 未排序 indices 触发 InvalidArgumentError
  - 重复 indices 触发 InvalidArgumentError
  - 维度不匹配触发 ValueError
  - 类型不匹配触发 TypeError
  - 负 shape 值触发 InvalidArgumentError

- 边界值（空、None、0 长度、极端形状/数值）：
  - 空稀疏张量 (nnz=0)
  - 全零密集张量转换
  - 极大形状 (接近 int64 上限)
  - 极端稀疏度 (nnz << prod(shape))
  - 浮点特殊值 (inf, nan, -0)

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow 运行时环境
  - 无外部网络/文件依赖

- 需要 mock/monkeypatch 的部分：
  - `tensorflow.python.ops.gen_sparse_ops` (底层操作)
  - `tensorflow.python.framework.ops` (图模式)
  - `tensorflow.python.eager.context` (执行上下文)

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 密集到稀疏转换验证索引排序
  2. 稀疏到密集转换验证值填充
  3. 稀疏矩阵乘法维度兼容性
  4. 边界条件处理（空张量、极端形状）
  5. 异常输入触发正确错误类型

- 可选路径（中/低优先级合并为一组列表）：
  - 不同数据类型组合测试
  - 跨设备（CPU/GPU）一致性
  - 图模式与 eager 模式等价性
  - 性能基准（大尺寸张量）
  - 内存使用验证

- 已知风险/缺失信息（仅列条目，不展开）：
  - 部分函数文档不完整
  - 边界条件细节缺失
  - 性能约束未明确
  - 异常处理细节不足