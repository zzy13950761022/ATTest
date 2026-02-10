# torch.cuda.random 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试 CUDA 随机数生成器状态管理、种子设置和查询功能，支持单 GPU 和多 GPU 环境
- 不在范围内的内容：非 CUDA 设备随机数生成、CPU 随机数生成器、第三方随机数库集成

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - `device`: int/str/torch.device，默认 'cuda'
  - `new_state`: torch.ByteTensor，无默认值
  - `seed`: int，无默认值
  - `new_states`: Iterable[Tensor]，无默认值
- 有效取值范围/维度/设备要求：
  - 设备索引：0 到 device_count-1
  - 状态张量：必须是 torch.ByteTensor 类型
  - 种子值：32位整数范围
- 必需与可选组合：
  - `get_rng_state`: 可选 device 参数
  - `set_rng_state`: 必需 new_state，可选 device
  - `manual_seed`: 必需 seed，可选 device
- 随机性/全局状态要求：
  - 状态张量必须保持连续内存格式
  - 多 GPU 环境需使用 `*_all` 函数保证确定性

## 3. 输出与判定
- 期望返回结构及关键字段：
  - `get_rng_state`: torch.ByteTensor（单设备状态）
  - `get_rng_state_all`: List[Tensor]（所有设备状态列表）
  - `set_rng_state/set_rng_state_all`: None
  - `manual_seed/manual_seed_all`: None
  - `seed/seed_all`: None
  - `initial_seed`: int（当前种子值）
- 容差/误差界（如浮点）：不适用（状态管理无数值计算）
- 状态变化或副作用检查点：
  - 调用 `get_rng_state` 或 `initial_seed` 会初始化 CUDA
  - 设置状态后随机数序列应保持一致
  - 多设备状态应独立管理

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非 ByteTensor 类型状态张量
  - 无效设备索引（超出范围）
  - 不兼容的状态张量形状/格式
  - 非整数种子值
- 边界值（空、None、0 长度、极端形状/数值）：
  - 空设备列表
  - None 作为状态张量
  - 极端种子值（0, -1, 2^31-1）
  - 空状态张量列表

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - CUDA 可用性（GPU 设备）
  - PyTorch CUDA 支持
  - 多 GPU 系统环境
- 需要 mock/monkeypatch 的部分：
  - CUDA 不可用场景
  - device_count 返回值
  - current_device 状态
  - 默认生成器管理

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 单设备状态获取与设置功能验证
  2. 种子设置与查询基本流程
  3. 多设备状态管理（*_all 函数）
  4. 无效设备索引异常处理
  5. 非 ByteTensor 状态张量类型检查
- 可选路径（中/低优先级合并为一组列表）：
  - 空状态张量处理
  - 极端种子值边界测试
  - 多线程并发访问安全性
  - 不同 PyTorch 版本状态兼容性
  - 内存格式连续性验证
  - 设备字符串标识符支持
- 已知风险/缺失信息（仅列条目，不展开）：
  - 多实体情况（模块包含8个函数）
  - 类型约束模糊（状态张量形状未明确）
  - 设备索引验证细节缺失
  - 状态张量版本兼容性未说明
  - 并发安全性未定义
  - 内存格式要求未在接口说明