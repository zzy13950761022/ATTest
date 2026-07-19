# torch.random - 函数说明

## 1. 基本信息
- **FQN**: torch.random
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/random.py`
- **签名**: 模块（包含多个函数）
- **对象类型**: module

## 2. 功能概述
`torch.random` 模块提供 PyTorch 随机数生成器的管理功能。包含设置/获取随机状态、种子管理、上下文管理等函数。主要用于控制随机数生成的可重复性。

## 3. 参数说明
模块包含以下核心函数：

**set_rng_state(new_state: torch.Tensor)**
- new_state (torch.ByteTensor): CPU RNG 状态张量

**get_rng_state() -> torch.Tensor**
- 无参数，返回 ByteTensor 状态

**manual_seed(seed) -> torch._C.Generator**
- seed (int): [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff] 范围内的整数

**seed() -> int**
- 无参数，返回 64 位随机种子

**initial_seed() -> int**
- 无参数，返回初始种子

**fork_rng(devices=None, enabled=True)**
- devices: CUDA 设备 ID 的可迭代对象（可选）
- enabled: 布尔值，控制是否启用（默认 True）

## 4. 返回值
- set_rng_state: None
- get_rng_state: torch.ByteTensor（RNG 状态）
- manual_seed: torch._C.Generator 对象
- seed: int（64 位随机数）
- initial_seed: int（初始种子值）
- fork_rng: 上下文管理器

## 5. 文档要点
- set_rng_state 仅适用于 CPU，CUDA 需用 manual_seed
- manual_seed 的 seed 范围：[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]
- 负种子通过公式 `0xffff_ffff_ffff_ffff + seed` 映射为正数
- fork_rng 默认操作所有 CUDA 设备，设备多时会警告

## 6. 源码摘要
- set_rng_state/get_rng_state: 调用 default_generator.set_state/get_state
- manual_seed: 处理种子转换，调用 CUDA manual_seed_all
- seed: 生成非确定性随机种子，同步 CUDA
- fork_rng: 使用 contextlib.contextmanager 实现 RNG 状态保存/恢复
- 依赖 torch._C.default_generator 核心生成器

## 7. 示例与用法（如有）
```python
# 设置随机种子
torch.random.manual_seed(42)

# 获取当前状态
state = torch.random.get_rng_state()

# 恢复状态
torch.random.set_rng_state(state)

# 使用上下文管理器
with torch.random.fork_rng():
    # 在此范围内的随机操作不影响外部状态
    pass
```

## 8. 风险与空白
- 模块包含多个函数实体，需分别测试
- set_rng_state 仅限 CPU 的约束需明确测试
- manual_seed 的种子范围边界条件需覆盖
- 负种子映射逻辑需验证
- fork_rng 的 devices 参数行为需测试
- 缺少 CUDA 特定函数的详细文档
- 未提供完整的错误处理示例
- 需要测试多设备环境下的行为