# torch.random 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试随机数生成器状态管理、种子设置、上下文隔离功能，确保随机操作可重复性
- 不在范围内的内容：具体随机数生成算法实现、第三方随机库兼容性、非PyTorch环境

## 2. 输入与约束
- 参数列表：
  - set_rng_state(new_state): ByteTensor类型CPU RNG状态
  - manual_seed(seed): int类型，范围[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]
  - fork_rng(devices, enabled): devices为CUDA设备ID可迭代对象，enabled为布尔值
- 有效取值范围/维度/设备要求：
  - set_rng_state仅限CPU张量
  - manual_seed负种子自动映射为正数
  - fork_rng默认操作所有CUDA设备
- 必需与可选组合：
  - get_rng_state/seed/initial_seed无参数
  - fork_rng的devices参数可选
- 随机性/全局状态要求：
  - 状态操作影响全局默认生成器
  - fork_rng创建临时隔离环境

## 3. 输出与判定
- 期望返回结构：
  - get_rng_state: ByteTensor状态张量
  - manual_seed: torch._C.Generator对象
  - seed/initial_seed: int类型64位值
  - set_rng_state: None
- 容差/误差界：无浮点容差要求
- 状态变化或副作用检查点：
  - set_rng_state后随机序列应恢复
  - manual_seed后随机序列应确定
  - fork_rng退出后外部状态应恢复

## 4. 错误与异常场景
- 非法输入/维度/类型：
  - set_rng_state传入非ByteTensor触发异常
  - manual_seed传入非整数类型触发异常
  - fork_rng传入非法设备ID触发异常
- 边界值：
  - manual_seed边界值：-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff
  - 空状态张量处理
  - fork_rng空设备列表
  - enabled=False时上下文行为

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - CUDA设备（可选，用于fork_rng测试）
  - 无网络/文件依赖
- 需要mock/monkeypatch的部分：
  - torch._C.default_generator（核心生成器）
  - torch.cuda.manual_seed_all（CUDA种子设置）
  - torch.cuda.device_count（设备数量检测）
  - torch.cuda.get_rng_state_all（多设备状态）

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. manual_seed正负边界值及负种子映射
  2. set_rng_state/get_rng_state状态保存恢复
  3. fork_rng上下文隔离与状态恢复
  4. seed函数生成非确定性随机数
  5. initial_seed返回初始种子值
- 可选路径（中/低优先级）：
  - 多CUDA设备下fork_rng行为
  - 连续调用seed的随机性
  - 混合CPU/CUDA操作状态同步
  - 大状态张量处理
  - 并发环境状态竞争
- 已知风险/缺失信息：
  - CUDA特定函数文档不完整
  - 多设备警告触发条件
  - 负种子映射公式验证
  - 状态张量内部结构未知