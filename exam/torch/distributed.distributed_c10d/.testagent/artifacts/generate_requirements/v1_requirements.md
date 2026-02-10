# torch.distributed.distributed_c10d 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试分布式通信核心模块，验证进程组管理、集体通信、点对点通信功能
- 不在范围内的内容：底层C++实现、网络协议细节、硬件特定优化

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - backend: str/Backend，必需，小写字符串（gloo/nccl/mpi/ucc）
  - init_method: str，可选，默认"env://"，与store互斥
  - world_size: int，store方式下必需，默认-1
  - rank: int，store方式下必需，默认-1，范围[0, world_size-1]
  - timeout: timedelta，默认1800秒
  - tensor: Tensor，必需，支持复数类型
  - op: ReduceOp，默认SUM，支持SUM/PRODUCT/MIN/MAX/BAND/BOR/BXOR
  - async_op: bool，默认False

- 有效取值范围/维度/设备要求：
  - NCCL后端要求每个进程独占GPU访问
  - 张量必须在相同设备上（CPU或GPU）
  - 进程组必须已初始化

- 必需与可选组合：
  - init_method与store参数互斥
  - store方式必须提供world_size和rank
  - 异步操作返回AsyncWork句柄

- 随机性/全局状态要求：
  - 全局进程组状态管理
  - 后端运行时检测

## 3. 输出与判定
- 期望返回结构及关键字段：
  - init_process_group：无返回值，初始化全局状态
  - all_reduce：async_op=True返回AsyncWork，否则None
  - 集体通信：原地修改输入张量

- 容差/误差界（如浮点）：
  - 浮点运算符合IEEE标准
  - 复数运算保持正确性

- 状态变化或副作用检查点：
  - 全局进程组状态更新
  - 张量内容正确归约
  - 异步操作状态跟踪

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 无效backend字符串
  - 张量类型不匹配
  - 进程组未初始化
  - 进程不在组内
  - init_method与store同时提供

- 边界值（空、None、0长度、极端形状/数值）：
  - world_size=0或负数
  - rank超出范围
  - 空张量或零维张量
  - 超时值为0或负数
  - 极端大张量（内存边界）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - CUDA环境（NCCL后端）
  - MPI编译支持
  - 网络连接（多机通信）
  - 共享存储（store初始化）

- 需要mock/monkeypatch的部分：
  - 进程组模拟
  - 网络通信
  - 后端可用性检测
  - 异步操作回调

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 基本进程组初始化与销毁
  2. all_reduce SUM操作正确性
  3. 多后端兼容性（至少gloo）
  4. 异步操作生命周期管理
  5. 错误参数异常触发

- 可选路径（中/低优先级合并为一组列表）：
  - 复数张量支持
  - 所有ReduceOp类型验证
  - 超时机制测试
  - 点对点通信验证
  - 多进程组管理
  - 内存边界测试
  - 性能基准测试

- 已知风险/缺失信息（仅列条目，不展开）：
  - 模块级docstring为空
  - 多实体模块测试复杂性
  - 分布式环境模拟困难
  - 后端行为差异
  - 异步操作返回值处理