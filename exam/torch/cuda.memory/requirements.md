# torch.cuda.memory 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证 GPU 内存管理模块的分配、监控、统计和清理功能，确保内存操作正确性和统计准确性
- 不在范围内的内容：底层 CUDA 驱动实现细节、非 CUDA 设备支持、第三方库 pynvml 的内部逻辑

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - `device`: torch.device/int/None，默认当前设备
  - `fraction`: float，范围 0~1
  - `size`: int，分配字节数
  - `mem_ptr`: int，内存指针地址
  - `abbreviated`: bool，默认 False
- 有效取值范围/维度/设备要求：
  - device 必须为有效 CUDA 设备索引或 torch.device('cuda')
  - fraction 必须在 [0, 1] 闭区间内
  - size 必须为正整数
- 必需与可选组合：device 参数通常可选，使用当前设备
- 随机性/全局状态要求：依赖 CUDA 运行时状态，测试需控制内存分配序列

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 内存统计：Dict[str, Any]，包含 allocated/reserved/active/inactive 等键
  - 内存信息：Tuple[int, int] (空闲内存, 总内存)
  - 操作函数：None 或 int 类型内存指针
  - 摘要信息：str 格式文本
- 容差/误差界（如浮点）：内存统计允许 ±1% 误差，浮点比较使用相对容差 1e-6
- 状态变化或副作用检查点：
  - empty_cache() 后缓存内存应减少
  - memory_allocated() 在分配后应增加，释放后应减少
  - 峰值统计应正确记录历史最大值

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 无效 device 参数触发 RuntimeError
  - fraction 超出 [0,1] 触发 ValueError
  - 非 CUDA 设备触发 RuntimeError
  - 内存不足时分配失败触发 RuntimeError
- 边界值（空、None、0 长度、极端形状/数值）：
  - device=None 使用当前设备
  - size=0 分配应失败或返回 None
  - fraction=0 或 fraction=1 的边界处理
  - 极大 size 值的内存分配失败

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - 需要 CUDA 兼容 GPU 设备
  - list_gpu_processes 依赖 pynvml 库
  - 需要 CUDA 运行时环境初始化
- 需要 mock/monkeypatch 的部分：
  - 模拟无 CUDA 设备环境
  - 模拟内存分配失败场景
  - 模拟 pynvml 库缺失或异常
  - 模拟不同 CUDA 版本行为

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基础内存分配与释放的正确统计
  2. empty_cache() 对缓存内存的影响验证
  3. 多设备环境下的内存操作隔离性
  4. 内存统计函数的嵌套字典结构完整性
  5. 异常参数输入的错误处理机制
- 可选路径（中/低优先级合并为一组列表）：
  - 大池(≥1MB)和小池(<1MB)分别统计准确性
  - memory_summary 格式化输出的完整性
  - 峰值统计在不同分配模式下的正确性
  - 并发环境下的内存操作安全性
  - 不同 CUDA 版本间的兼容性
- 已知风险/缺失信息（仅列条目，不展开）：
  - 多线程/多进程并发安全未明确
  - 部分函数缺少返回类型注解
  - 内存分配失败的具体错误信息格式
  - 极端内存压力下的行为定义
  - 平台特定行为差异文档不足