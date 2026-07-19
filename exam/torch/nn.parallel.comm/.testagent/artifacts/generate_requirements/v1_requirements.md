# torch.nn.parallel.comm 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证多GPU通信模块的广播、散射、聚集和归约操作正确性，确保分布式训练数据并行通信可靠
- 不在范围内的内容：底层C++实现细节、NCCL后端内部逻辑、性能基准测试、大规模集群通信

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - broadcast: tensor(Tensor), devices(Iterable[device], optional), out(Sequence[Tensor], optional)
  - broadcast_coalesced: tensors(sequence), devices(Iterable[device]), buffer_size(int=10485760)
  - reduce_add: inputs(Iterable[Tensor]), destination(int, optional)
  - reduce_add_coalesced: inputs(Iterable[Iterable[Tensor]]), destination(int, optional), buffer_size(int=10485760)
  - scatter: tensor(Tensor), devices(Iterable[device], optional), chunk_sizes(Iterable[int], optional), dim(int=0), streams(Iterable[Stream], optional), out(Sequence[Tensor], optional)
  - gather: tensors(Iterable[Tensor]), dim(int=0), destination(device, optional), out(Tensor, optional)

- 有效取值范围/维度/设备要求：
  - 张量必须在GPU上（reduce_add明确要求）
  - broadcast_coalesced要求所有张量在同一设备上
  - 设备参数支持torch.device、str或int格式
  - 输入张量形状、dtype、布局必须匹配（reduce_add）
  - dim参数必须在张量维度范围内

- 必需与可选组合：
  - broadcast: devices和out必须二选一
  - scatter: devices和out必须二选一
  - gather: destination和out互斥
  - chunk_sizes总和必须等于输入张量dim维度大小

- 随机性/全局状态要求：无随机性要求，依赖GPU设备状态

## 3. 输出与判定
- 期望返回结构及关键字段：
  - broadcast/broadcast_coalesced/scatter: 返回包含输出张量的元组
  - reduce_add/reduce_add_coalesced: 返回位于目标设备上的求和张量（或元组）
  - gather: 返回位于目标设备上的连接张量
  - 输出张量数量与目标设备数量一致

- 容差/误差界（如浮点）：浮点运算误差在1e-6范围内

- 状态变化或副作用检查点：
  - 输出张量设备正确性
  - 数据一致性（广播后数据相同，聚集后数据完整）
  - 内存不泄漏（GPU内存使用合理）

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非GPU张量输入（reduce_add）
  - 设备列表为空或无效
  - 形状不匹配（reduce_add）
  - 参数组合冲突（devices/out同时提供）
  - 维度超出范围
  - chunk_sizes总和与输入维度不匹配

- 边界值（空、None、0长度、极端形状/数值）：
  - 空设备列表
  - 空张量序列
  - 零维张量
  - 极端大形状张量（内存边界）
  - 极端小缓冲区大小
  - 复数张量处理

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - 至少2个可用GPU设备
  - CUDA环境
  - PyTorch安装（包含C++扩展）
  - 可选：NCCL后端（如果测试分布式特性）

- 需要mock/monkeypatch的部分：
  - GPU设备不可用时的降级行为
  - 底层C++函数调用
  - 设备索引转换函数
  - 复数张量处理逻辑

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 单GPU到多GPU广播功能正确性
  2. 多GPU张量归约求和数值正确性
  3. 张量分散-聚集往返数据完整性
  4. 参数组合冲突的异常处理
  5. 设备边界和无效输入的错误处理

- 可选路径（中/低优先级合并为一组列表）：
  - 缓冲区大小对性能的影响
  - 复数张量支持
  - 流同步行为
  - 稀疏张量处理
  - 大规模张量内存管理
  - 混合精度支持
  - 跨设备类型兼容性

- 已知风险/缺失信息（仅列条目，不展开）：
  - 底层C++实现细节未知
  - 设备兼容性约束不明确
  - 错误处理边界条件未完全文档化
  - 流同步行为未明确文档化
  - 性能特性未详细说明