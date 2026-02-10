# tensorflow.python.ops.script_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证Python函数包装为TensorFlow操作的正确性
  - 测试eager执行和graph模式下函数调用一致性
  - 验证NumPy数组与Tensor之间的转换逻辑
  - 确保函数注册和调用机制正常工作
- 不在范围内的内容
  - XLA编译支持（明确不支持）
  - SavedModel序列化（函数体不保存）
  - 分布式跨进程执行（必须在同一进程）
  - 性能基准测试（GIL限制已知）

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - func: callable，必需，接受Tensor/NumPy数组，返回Tensor/NumPy数组
  - inp: list[Tensor/CompositeTensor]，必需，长度匹配func参数
  - Tout: list/tf.DType/tf.TypeSpec，必需，指定返回类型
  - stateful: bool，默认True，仅py_func_common
  - name: str/None，可选，操作名称
- 有效取值范围/维度/设备要求
  - func必须与调用程序在同一地址空间
  - inp元素必须是Tensor或CompositeTensor
  - Tout必须与func返回值类型匹配
  - 不支持jit_compile=True
- 必需与可选组合
  - func、inp、Tout为必需参数
  - name为可选参数
  - stateful仅适用于py_func_common变体
- 随机性/全局状态要求
  - stateful=True时函数可能有副作用
  - 函数调用会获取Python GIL
  - 不支持异步执行

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回Tensor、CompositeTensor或列表
  - 返回类型由Tout参数确定
  - func返回None时返回空列表
  - 返回值形状与func输出一致
- 容差/误差界（如浮点）
  - NumPy数组转换使用默认精度
  - 浮点运算遵循TensorFlow标准容差
  - 类型转换遵循TensorFlow类型提升规则
- 状态变化或副作用检查点
  - 验证FuncRegistry注册正确性
  - 检查设备间复制逻辑（_maybe_copy_to_context_device）
  - 验证CompositeTensor包装处理（_wrap_for_composites）

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - func不是callable时抛出TypeError
  - inp不是list时抛出TypeError
  - Tout与func返回值类型不匹配时抛出异常
  - 尝试使用jit_compile=True时抛出错误
- 边界值（空、None、0长度、极端形状/数值）
  - inp为空列表（func无参数）
  - func返回None（应返回空列表）
  - 输入Tensor形状为0或极端维度
  - CompositeTensor边界情况
  - 嵌套结构输入输出

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - Python解释器环境（同一进程）
  - TensorFlow运行时
  - NumPy库（py_func变体）
  - 设备上下文（CPU/GPU）
- 需要mock/monkeypatch的部分
  - `tensorflow.python.framework.func_graph.FuncGraph`（graph模式）
  - `tensorflow.python.eager.context.context`（eager执行）
  - `tensorflow.python.framework.ops.get_default_graph`
  - `tensorflow.python.ops.script_ops._internal_py_func`（核心实现）
  - `tensorflow.python.ops.script_ops.FuncRegistry`（函数注册）
  - `tensorflow.python.ops.script_ops._maybe_copy_to_context_device`
  - `tensorflow.python.ops.script_ops._wrap_for_composites`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. eager_py_func基础调用：Tensor输入输出
  2. py_func_common：NumPy数组转换
  3. numpy_function别名功能验证
  4. 复合张量（CompositeTensor）支持
  5. 错误处理：类型不匹配、无效输入
- 可选路径（中/低优先级合并为一组列表）
  - 多返回值场景（Tout为列表）
  - stateful参数影响（True/False）
  - 嵌套函数调用
  - 设备间数据复制
  - 梯度计算支持（EagerFunc）
  - 函数注册表管理（FuncRegistry）
  - 名称参数使用
  - 空输入/输出边界
- 已知风险/缺失信息（仅列条目，不展开）
  - GIL获取对并发性能影响
  - 函数体不序列化到GraphDef
  - 不支持跨进程分布式
  - 缺少详细错误消息文档
  - 异步执行限制未明确