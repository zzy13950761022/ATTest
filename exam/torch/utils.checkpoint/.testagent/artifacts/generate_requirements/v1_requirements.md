# torch.utils.checkpoint 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证梯度检查点机制正确性，确保前向传播节省内存，反向传播梯度计算准确
- 不在范围内的内容：`checkpoint_sequential`辅助函数、自定义检查点实现、性能基准测试

## 2. 输入与约束
- 参数列表：
  - `function` (callable)：可调用对象，接受`*args`参数
  - `*args`：任意位置参数元组，传递给function
  - `use_reentrant` (bool, 默认True)：梯度计算模式开关
  - `**kwargs`：仅当`use_reentrant=False`时支持的关键字参数
  - `preserve_rng_state` (bool, 默认True)：通过kwargs传递的RNG状态控制

- 有效取值范围/维度/设备要求：
  - function必须为可调用对象，前向/反向行为一致
  - 输入Tensor支持CPU/CUDA设备
  - 输出可包含非Tensor值，仅Tensor参与梯度计算

- 必需与可选组合：
  - function为必需参数
  - use_reentrant=True时，kwargs不支持
  - use_reentrant=False时，支持kwargs传递

- 随机性/全局状态要求：
  - preserve_rng_state=True时，必须保存/恢复CPU和CUDA RNG状态
  - 随机操作在检查点前后必须产生相同结果

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 返回类型与function(*args)输出完全一致
  - 支持Tensor、元组、列表、字典等任意嵌套结构
  - 非Tensor值原样返回，不参与梯度计算

- 容差/误差界：
  - 浮点误差：与直接运行function相比，结果差异在1e-6范围内
  - 梯度误差：反向传播梯度与直接计算梯度差异在1e-5范围内

- 状态变化或副作用检查点：
  - 验证RNG状态在检查点前后保持一致
  - 确保自动梯度上下文正确管理
  - 内存使用量应低于直接运行function

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常：
  - function不可调用：TypeError
  - use_reentrant=True时传递kwargs：RuntimeError
  - 前向/反向行为不一致：RuntimeError
  - 嵌套Tensor结构处理不当：RuntimeError

- 边界值测试：
  - 空参数列表：function()
  - None作为function参数
  - 0维Tensor输入
  - 极端形状：大尺寸Tensor(>1GB)
  - 混合设备输入(CPU+CUDA)
  - 包含requires_grad=False的Tensor

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - CUDA设备（可选，用于GPU测试）
  - torch.autograd.backward/grad函数
  - 随机数生成器状态管理

- 需要mock/monkeypatch的部分：
  - `torch.utils.checkpoint.CheckpointFunction.apply`
  - `torch.utils.checkpoint._checkpoint_without_reentrant`
  - `torch.utils.checkpoint.detach_variable`
  - `torch.utils.checkpoint.check_backward_validity`
  - `torch.utils.checkpoint.get_device_states`
  - `torch.utils.checkpoint.set_device_states`
  - `torch.cuda.get_rng_state` / `torch.cuda.set_rng_state`
  - `torch.get_rng_state` / `torch.set_rng_state`

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. use_reentrant=True模式基础功能验证
  2. use_reentrant=False模式关键字参数支持
  3. 梯度正确性：与直接计算对比
  4. RNG状态保存/恢复机制
  5. 内存使用量验证

- 可选路径（中/低优先级）：
  - 嵌套输出结构处理（列表/字典中的Tensor）
  - 混合设备输入场景
  - 大尺寸Tensor内存边界测试
  - 多次嵌套checkpoint调用
  - 自定义autograd.Function作为function
  - 多线程/多进程环境

- 已知风险/缺失信息：
  - 模块级导入路径不明确
  - 类型注解不完整
  - CUDA状态管理细节未文档化
  - 嵌套Tensor结构处理边界
  - 自定义对象序列化支持
  - 性能开销量化标准缺失