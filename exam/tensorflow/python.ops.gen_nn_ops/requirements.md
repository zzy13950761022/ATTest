# tensorflow.python.ops.gen_nn_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证神经网络操作函数（AvgPool, Conv2D, BatchNorm等）的正确执行
  - 确保参数验证、类型检查和形状计算符合TensorFlow规范
  - 验证不同数据格式（NHWC/NCHW）下的计算结果一致性
  - 确保梯度计算和反向传播正常工作
- 不在范围内的内容
  - 单个函数的极端性能优化测试
  - TensorFlow框架本身的底层C++实现验证
  - 与其他深度学习框架的交叉兼容性测试

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - value: 4-D Tensor, 支持 half/bfloat16/float32/float64
  - ksize: list[int], 长度≥4, 无默认值
  - strides: list[int], 长度≥4, 无默认值
  - padding: string, 可选"SAME"/"VALID", 无默认值
  - data_format: string, 默认"NHWC", 可选"NHWC"/"NCHW"
  - name: string, 可选, 默认None
- 有效取值范围/维度/设备要求
  - 输入张量必须是4维：[batch, height, width, channels]
  - ksize和strides长度必须≥4
  - 支持CPU和GPU设备
  - 数值范围受浮点类型限制
- 必需与可选组合
  - value, ksize, strides, padding为必需参数
  - data_format和name为可选参数
- 随机性/全局状态要求
  - 无随机性要求
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回与输入value相同类型的Tensor
  - 输出形状由输入形状、ksize、strides和padding决定
  - 对于"SAME" padding: output_shape = ceil(input_shape / stride)
  - 对于"VALID" padding: output_shape = ceil((input_shape - ksize + 1) / stride)
- 容差/误差界（如浮点）
  - float32: 相对误差≤1e-5
  - float64: 相对误差≤1e-10
  - half/bfloat16: 相对误差≤1e-2
- 状态变化或副作用检查点
  - 无文件I/O或网络操作
  - 无全局状态修改
  - 计算图构建正确性

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 非4维输入张量 → ValueError
  - ksize/strides长度<4 → ValueError
  - 无效padding值 → ValueError
  - 无效data_format值 → ValueError
  - 不支持的数据类型 → TypeError
- 边界值（空、None、0长度、极端形状/数值）
  - batch_size=0或1的边界情况
  - ksize=[1,1,1,1]的最小窗口
  - strides=[1,1,1,1]的最小步长
  - 极端大形状（内存边界）
  - NaN/Inf输入值处理

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow Python包（≥2.0.0）
  - 可选GPU支持（CUDA/cuDNN）
  - 无网络或文件系统依赖
- 需要mock/monkeypatch的部分
  - `tensorflow.python.eager.context`（用于设备上下文）
  - `tensorflow.python.framework.ops.get_default_graph`
  - `tensorflow.python.ops.gradients_impl.gradients`
  - `pywrap_tfe.TFE_Py_FastPathExecute`（快速执行路径）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. 4维浮点张量的基本池化操作
  2. NHWC和NCHW数据格式的结果一致性
  3. "SAME"和"VALID" padding的正确形状计算
  4. 梯度计算和反向传播验证
  5. 参数验证和异常处理
- 可选路径（中/低优先级合并为一组列表）
  - 不同浮点精度（half/bfloat16/float32/float64）的数值稳定性
  - 极端形状和边界条件的鲁棒性
  - 批量大小变化对性能的影响
  - 不同设备（CPU/GPU）的结果一致性
  - 内存使用和泄漏检查
- 已知风险/缺失信息（仅列条目，不展开）
  - 机器生成代码的文档不完整性
  - 某些参数的具体约束未明确说明
  - 多函数模块的测试选择策略
  - 版本兼容性风险