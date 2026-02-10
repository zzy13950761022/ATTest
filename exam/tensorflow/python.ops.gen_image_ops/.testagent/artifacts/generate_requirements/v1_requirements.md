# tensorflow.python.ops.gen_image_ops 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证约50个图像处理函数的正确性，包括图像编解码、颜色空间转换、尺寸调整、裁剪、边界框处理等操作
- 不在范围内的内容：不测试TensorFlow底层C++实现，不覆盖所有50个函数的完整组合测试，不测试性能指标

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - 图像张量：3D/4D张量，shape=[batch, height, width, channels]或[height, width, channels]
  - 尺寸参数：int32张量，指定输出高宽
  - 阈值参数：float32张量，用于非极大值抑制等操作
  - 可选参数：插值方法（bilinear/nearest等）、对齐方式、填充模式
- 有效取值范围/维度/设备要求：
  - 图像至少3D，最后3维为[height, width, channels]
  - 支持数据类型：uint8, float32, float64等（各函数不同）
  - 设备：CPU/GPU均可，依赖TensorFlow运行时
- 必需与可选组合：各函数参数组合不同，需按函数文档测试
- 随机性/全局状态要求：无全局状态依赖，部分函数有随机性（如随机裁剪）

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 图像处理函数：返回处理后的张量，保持输入数据类型
  - 编解码函数：返回uint8或指定类型的张量
  - 边界框函数：返回索引或坐标张量
  - 复合函数：返回命名元组包含多个张量
- 容差/误差界（如浮点）：
  - 浮点运算：相对误差1e-5，绝对误差1e-8
  - 图像处理：像素值误差在合理范围内
- 状态变化或副作用检查点：无副作用，纯函数操作

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 维度不足（<3D）触发ValueError
  - 不支持的数据类型触发TypeError
  - 无效参数值触发InvalidArgumentError
- 边界值（空、None、0长度、极端形状/数值）：
  - 空张量或零尺寸图像
  - 极端形状（超大/超小尺寸）
  - 边界数值（0, 255, -inf, inf, nan）
  - None输入参数

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - TensorFlow运行时环境
  - 图像编解码依赖系统库（libjpeg, libpng等）
  - GPU设备（可选）
- 需要mock/monkeypatch的部分：
  - `tensorflow.python.framework.ops.get_default_graph`
  - `tensorflow.python.eager.context.context`
  - `tensorflow.python.framework.dtypes.as_dtype`
  - `tensorflow.python.ops.gen_image_ops._op_def_library._apply_op_helper`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. decode_jpeg/decode_png编解码正确性
  2. resize_bilinear/resize_nearest_neighbor尺寸调整
  3. non_max_suppression边界框处理
  4. adjust_contrastv2/adjust_brightness颜色调整
  5. crop_and_resize裁剪与调整
- 可选路径（中/低优先级合并为一组列表）：
  - 其他图像变换函数（rotate, flip, transpose）
  - 颜色空间转换（rgb_to_hsv, hsv_to_rgb）
  - 图像统计函数（image_gradients, sobel_edges）
  - 特殊处理函数（extract_glimpse, draw_bounding_boxes）
- 已知风险/缺失信息（仅列条目，不展开）：
  - 机器生成代码可能被覆盖
  - 依赖TensorFlow C++后端实现
  - 约50个函数测试覆盖不完整
  - 缺少模块级使用示例
  - 各函数边界条件需单独分析