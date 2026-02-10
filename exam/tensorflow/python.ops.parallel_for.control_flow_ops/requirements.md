# tensorflow.python.ops.parallel_for.control_flow_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证 for_loop、pfor、vectorized_map 三个函数的并行循环控制流
  - 确保向量化循环正确执行迭代计算
  - 验证张量嵌套结构的正确展开和重组
  - 测试 CompositeTensor（SparseTensor、IndexedSlices）支持
  - 验证并行迭代控制（parallel_iterations）的内存管理
- 不在范围内的内容
  - 不支持迭代间数据依赖的复杂场景
  - 不支持 tf.cond 等复杂控制流
  - 不测试 XLA 编译器的内部优化细节
  - 不验证底层 PFor 转换器的实现细节

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - for_loop: loop_fn(函数), loop_fn_dtypes(dtypes), iters(int), parallel_iterations(可选int)
  - pfor: loop_fn(函数), iters(int), fallback_to_while_loop(bool), parallel_iterations(可选int, >1)
  - vectorized_map: fn(函数), elems(张量/嵌套结构), fallback_to_while_loop(bool)
- 有效取值范围/维度/设备要求
  - iters: 非负整数，支持 0 和正整数
  - parallel_iterations: 正整数，pfor 必须 >1
  - loop_fn 输出形状不应依赖输入
  - elems 必须沿第一维展开
- 必需与可选组合
  - for_loop: loop_fn、loop_fn_dtypes、iters 必需，parallel_iterations 可选
  - pfor: loop_fn、iters 必需，fallback_to_while_loop 可选，parallel_iterations 可选
  - vectorized_map: fn、elems 必需，fallback_to_while_loop 可选
- 随机性/全局状态要求
  - 支持 RandomFoo 操作的状态读取
  - 支持 Variable 读取操作
  - 不支持跨迭代的状态修改

## 3. 输出与判定
- 期望返回结构及关键字段
  - for_loop/pfor: 与 loop_fn 输出相同结构的堆叠张量
  - vectorized_map: 与 fn 输出相同结构的堆叠张量
  - 输出维度：第一维为迭代次数，后续维度与 loop_fn/fn 输出一致
- 容差/误差界（如浮点）
  - 浮点计算误差在 1e-6 范围内
  - 并行与顺序执行结果应一致（误差范围内）
- 状态变化或副作用检查点
  - 验证无意外变量修改
  - 检查 RandomFoo 操作的正确状态管理
  - 验证 Variable 读取的正确性

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - iters 为负数触发 ValueError
  - parallel_iterations <= 0 触发 ValueError
  - pfor 中 parallel_iterations=1 触发 ValueError
  - loop_fn 输出形状依赖输入触发错误
  - 不支持的控制流操作触发 NotImplementedError
- 边界值（空、None、0 长度、极端形状/数值）
  - iters=0 返回空结构
  - 空张量输入处理
  - 极大迭代次数（内存边界）
  - 极端形状张量（高维、大尺寸）
  - parallel_iterations 超过 iters 的处理

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow 运行时环境
  - GPU 设备可用性（可选测试）
  - 足够内存处理大张量
- 需要 mock/monkeypatch 的部分
  - `tensorflow.python.ops.parallel_for.pfor.PFor` 转换器
  - `tensorflow.python.ops.control_flow_ops.while_loop` 回退机制
  - `tensorflow.python.ops.tensor_array_ops.TensorArray` 实现
  - `tensorflow.python.framework.composite_tensor.CompositeTensor` 处理

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. for_loop 基础功能：标量、向量、矩阵迭代
  2. pfor 向量化转换：简单算术运算的并行化
  3. vectorized_map 广播功能：不同形状输入处理
  4. CompositeTensor 支持：SparseTensor、IndexedSlices
  5. fallback_to_while_loop 机制：向量化失败回退
- 可选路径（中/低优先级合并为一组列表）
  - 嵌套结构输入输出处理
  - 并行迭代数优化测试
  - 内存使用边界测试
  - 随机操作状态管理
  - 变量读取操作验证
  - 零输出 Operation 对象支持
  - 不同设备（CPU/GPU）行为一致性
- 已知风险/缺失信息（仅列条目，不展开）
  - 实验性功能，API 可能变更
  - 不支持迭代间数据依赖
  - 复杂控制流支持有限
  - 缺少完整类型注解
  - XLA 上下文行为未充分测试