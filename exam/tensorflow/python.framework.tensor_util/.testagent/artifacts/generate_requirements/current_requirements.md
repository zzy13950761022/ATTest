# tensorflow.python.framework.tensor_util 测试需求

## 1. 目标与范围
- 验证 `make_tensor_proto` 正确转换 Python 数据为 TensorProto
- 测试数据类型推断、形状验证、广播功能
- 验证与 numpy 数组的双向转换兼容性
- 不包含：TensorFlow 2.0 新 API、分布式环境、GPU 设备

## 2. 输入与约束
- values: Python 标量/列表/numpy 数组/numpy 标量/TensorProto
- dtype: tensor_pb2.DataType 枚举值，可选，默认 None（自动推断）
- shape: 整数列表，可选，默认 None（自动推断）
- verify_shape: 布尔值，默认 False，启用形状验证
- allow_broadcast: 布尔值，默认 False，允许标量和长度1向量广播
- 约束：verify_shape 和 allow_broadcast 互斥
- 设备要求：仅 CPU，不依赖 GPU 或 TPU

## 3. 输出与判定
- 返回 TensorProto 对象，包含序列化张量数据
- 输入为 TensorProto 时直接返回原对象
- 浮点类型容差：相对误差 1e-6，绝对误差 1e-8
- 状态变化：无全局状态修改，纯函数
- 副作用检查：无文件/网络操作，无缓存

## 4. 错误与异常场景
- 非法类型：非数值/非列表/非数组输入
- 形状不匹配：verify_shape=True 时形状不一致
- 广播冲突：allow_broadcast=False 时标量/长度1向量
- 数据类型不兼容：值范围超出 dtype 表示范围
- 边界值：空列表、None 值、0 长度数组
- 极端形状：超大维度（>8）、超大尺寸（内存限制）
- 极端数值：NaN、Inf、极大/极小浮点数

## 5. 依赖与环境
- 外部依赖：numpy、tensor_pb2、dtypes、fast_tensor_util
- 需要 mock：fast_tensor_util 不可用时的降级路径
- 环境要求：Python 3.7+，TensorFlow 1.x/2.x 兼容模式
- 文件依赖：无
- 网络依赖：无

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. 基本数据类型转换（int32, float32, bool, string）
  2. 形状验证与广播功能互斥性
  3. 特殊数据类型支持（float16, bfloat16, complex）
  4. TensorProto 输入直接返回
  5. 空值和边界形状处理

- 可选路径（中/低优先级）：
  - 多维数组转换性能
  - 大尺寸张量内存限制
  - 与 MakeNdarray 的双向转换一致性
  - 不同 numpy 版本兼容性
  - 遗留 TensorFlow 1.x 工作流

- 已知风险/缺失信息：
  - fast_tensor_util 降级处理未文档化
  - 异常消息格式不统一
  - 内存使用峰值未定义
  - 并发调用安全性未验证