# tensorflow.python.ops.numerics 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证张量不包含 NaN 或 Inf 值，检查失败时记录错误消息，返回带数值检查依赖的原始张量
- 不在范围内的内容：不修改张量数值，不处理非浮点数据类型，不兼容 eager execution 的 `add_check_numerics_ops` 函数

## 2. 输入与约束
- 参数列表：
  - x (Tensor)：任意形状的浮点张量，自动转换为 Tensor 类型
  - message (str)：检查失败时的错误消息字符串
  - name (str/None)：操作名称，可选，默认 None
- 有效取值范围/维度/设备要求：浮点数据类型（float16, float32, float64），任意形状，与输入张量同设备执行
- 必需与可选组合：x 和 message 必需，name 可选
- 随机性/全局状态要求：无随机性，依赖 TensorFlow 计算图执行

## 3. 输出与判定
- 期望返回结构及关键字段：与输入张量 x 相同的 Tensor 对象，添加数值检查依赖
- 容差/误差界（如浮点）：严格检查 NaN/Inf，无容差
- 状态变化或副作用检查点：检查失败时记录错误消息，确保检查依赖在张量使用前执行

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：非张量输入、非字符串 message、不支持的数据类型
- 边界值（空、None、0 长度、极端形状/数值）：空张量、零维张量、包含 NaN/Inf 的张量、极大/极小浮点值

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：TensorFlow 运行时，GPU/CPU 设备支持
- 需要 mock/monkeypatch 的部分：
  - `tensorflow.python.ops.array_ops.check_numerics`
  - `tensorflow.python.ops.control_flow_ops.with_dependencies`
  - `tensorflow.python.framework.ops.convert_to_tensor`
  - `tensorflow.python.framework.ops.colocate_with`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 正常浮点张量无 NaN/Inf 通过检查
  2. 包含 NaN 的张量触发错误记录
  3. 包含 Inf 的张量触发错误记录
  4. 不同浮点数据类型（float16/32/64）的兼容性
  5. 不同形状张量（标量、向量、矩阵、高维）的检查
- 可选路径（中/低优先级合并为一组列表）：
  - v1 版本函数 `verify_tensor_all_finite` 的别名参数兼容性
  - 空张量和零维张量的边界情况
  - 与非浮点数据类型（int, bool）的交互
  - `add_check_numerics_ops` 函数的控制流限制
  - 操作名称参数 name 的功能验证
- 已知风险/缺失信息（仅列条目，不展开）：
  - v1 版本参数别名复杂性
  - `add_check_numerics_ops` 不兼容 eager execution
  - 未明确指定所有支持的数据类型
  - 错误消息记录的具体格式未定义