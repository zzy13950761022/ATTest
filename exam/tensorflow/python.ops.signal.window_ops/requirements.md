# tensorflow.python.ops.signal.window_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证5个窗口函数（kaiser_window, kaiser_bessel_derived_window, vorbis_window, hann_window, hamming_window）正确生成指定长度的窗口张量
- 不在范围内的内容：窗口函数的数学理论推导、与其他库（scipy/numpy）的完全一致性、实时性能基准

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - window_length: int/Tensor (rank 0标量)，无默认值（必需）
  - beta: float/Tensor (kaiser系列)，默认12.0
  - periodic: bool/Tensor (hann/hamming)，默认True
  - dtype: DType，默认float32
  - name: str，默认None
- 有效取值范围/维度/设备要求：
  - window_length ≥ 1（整数）
  - dtype必须是浮点类型（float16/float32/float64）
  - 支持CPU/GPU设备
  - beta参数无明确范围限制
- 必需与可选组合：
  - window_length为必需参数
  - beta仅kaiser系列需要
  - periodic仅hann/hamming需要
- 随机性/全局状态要求：无随机性，无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 形状为[window_length]的一维Tensor
  - 数据类型与指定dtype一致
  - 窗口值在[0, 1]范围内（部分窗口可能超出）
- 容差/误差界（如浮点）：
  - float32: 相对误差1e-5，绝对误差1e-6
  - float64: 相对误差1e-10，绝对误差1e-12
  - float16: 允许较大误差（1e-2级别）
- 状态变化或副作用检查点：
  - 无文件I/O
  - 无全局状态修改
  - 无随机数生成器影响

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - window_length非标量：ValueError
  - window_length ≤ 0：InvalidArgumentError
  - dtype非浮点类型：TypeError
  - beta非数值类型：TypeError
  - periodic非布尔类型：TypeError
- 边界值（空、None、0长度、极端形状/数值）：
  - window_length=1：返回单元素张量
  - 极大window_length（>1e6）：内存/性能测试
  - beta极端值（0, 1e6）：数值稳定性
  - dtype=float16：精度损失验证

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow运行时
  - 无网络/文件系统依赖
- 需要mock/monkeypatch的部分：
  - `tensorflow.python.ops.math_ops`（数值计算）
  - `tensorflow.python.ops.array_ops`（张量操作）
  - `tensorflow.python.ops.special_math_ops`（特殊函数）
  - `tensorflow.python.framework.dtypes`（类型系统）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 基本功能：各窗口函数生成正确形状和类型
  2. 边界条件：window_length=1和极大值处理
  3. 参数验证：非法dtype和window_length触发异常
  4. 精度验证：不同dtype下的数值稳定性
  5. 设备兼容：CPU/GPU一致输出
- 可选路径（中/低优先级合并为一组列表）：
  - periodic参数对hann/hamming窗口的影响
  - beta参数对kaiser窗口形状的影响
  - 与numpy/scipy参考实现的数值比较
  - 梯度计算正确性（如果支持）
  - 批处理输入支持情况
- 已知风险/缺失信息（仅列条目，不展开）：
  - beta参数有效范围未定义
  - float16精度损失具体界限
  - 窗口函数的数值稳定性证明
  - 内存使用峰值未量化