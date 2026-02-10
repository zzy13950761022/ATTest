# torch.autograd.profiler 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证 profile 上下文管理器正确记录 PyTorch 操作执行事件和时间统计，支持多种分析模式配置
- 不在范围内的内容：分布式分析、第三方分析工具集成、非 PyTorch 操作分析

## 2. 输入与约束
- 参数列表：
  - enabled: bool, 默认 True
  - use_cuda: bool, 默认 False
  - record_shapes: bool, 默认 False
  - with_flops: bool, 默认 False
  - profile_memory: bool, 默认 False
  - with_stack: bool, 默认 False
  - with_modules: bool, 默认 False
  - use_kineto: bool, 默认 False
  - use_cpu: bool, 默认 True
  - experimental_config: _ExperimentalConfig, 默认 None

- 有效取值范围/维度/设备要求：
  - use_cuda=True 需要 CUDA 可用设备
  - with_flops=True 仅支持矩阵乘和 2D 卷积操作
  - with_modules=True 仅支持 TorchScript 模型
  - 不支持嵌套 profile 调用

- 必需与可选组合：
  - use_cpu=True 或 use_cuda=True 至少一个为 True
  - 所有 bool 参数可任意组合

- 随机性/全局状态要求：
  - 分析器是线程本地的
  - 修改全局分析器状态
  - 使用 CUDA 时调用 torch.cuda.synchronize()

## 3. 输出与判定
- 期望返回结构及关键字段：
  - profile 实例包含 EventList 对象
  - table() 方法返回格式化的时间统计表
  - key_averages() 返回聚合事件统计
  - total_average() 返回总体平均统计
  - export_chrome_trace() 返回 JSON 格式跟踪数据

- 容差/误差界：
  - CPU 时间测量误差 < 1ms
  - CUDA 事件时间误差 < 0.1ms
  - FLOPs 估计误差 < 5%

- 状态变化或副作用检查点：
  - 分析结束后全局分析器状态恢复
  - 无内存泄漏（profile_memory=True 时）
  - 无 CUDA 上下文污染

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常：
  - use_cuda=True 但无 CUDA 设备时抛出 RuntimeError
  - 嵌套 profile 调用时抛出 RuntimeError
  - use_cuda=True 且 DataLoader num_workers>0 时警告
  - 无效 experimental_config 类型时抛出 TypeError

- 边界值：
  - enabled=False 时无分析数据
  - 空操作序列（无 autograd 操作）
  - 极端形状张量（如 0 维、超大形状）
  - 极端数值（inf, nan, 极大/极小值）
  - 混合精度操作

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - CUDA 设备（use_cuda=True 时）
  - 文件系统（export_chrome_trace 时）
  - Kineto 库（use_kineto=True 时）

- 需要 mock/monkeypatch 的部分：
  - torch.cuda.synchronize() 调用
  - _enable_profiler/_disable_profiler C++ 后端
  - 文件 I/O 操作
  - CUDA 设备可用性检查
  - 时间测量函数

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. 基本 CPU 分析：enabled=True, use_cpu=True 默认配置
  2. CUDA 分析：use_cuda=True 且 CUDA 可用
  3. 形状记录：record_shapes=True 验证张量形状捕获
  4. 内存分析：profile_memory=True 验证内存跟踪
  5. 嵌套操作检测：验证非可重入性约束

- 可选路径（中/低优先级）：
  - with_flops=True 的 FLOPs 估计
  - with_stack=True 的源代码归属
  - with_modules=True 的模块层次记录
  - use_kineto=True 的 Kineto 集成
  - experimental_config 实验配置
  - 异步任务中的分析器传播
  - 多线程环境下的线程本地行为
  - 导出 Chrome 跟踪文件格式验证

- 已知风险/缺失信息：
  - 实验性配置选项文档不完整
  - 分布式分析支持有限
  - 异步事件处理逻辑复杂
  - 部分类型注解缺失
  - CUDA 分析对 DataLoader 的限制
  - 形状记录可能引入分析偏差