# torch.cuda.memory - 函数说明

## 1. 基本信息
- **FQN**: torch.cuda.memory
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/cuda/memory.py`
- **签名**: 模块（包含多个函数）
- **对象类型**: module

## 2. 功能概述
`torch.cuda.memory` 是 PyTorch CUDA 内存管理模块，提供 GPU 内存分配、监控和统计功能。包含内存分配器操作、内存统计查询、缓存管理和进程监控等工具函数。

## 3. 参数说明
模块包含多个函数，主要参数模式：
- `device` (torch.device/int/None): 目标 GPU 设备，默认当前设备
- `fraction` (float): 内存比例，范围 0~1
- `size` (int): 分配字节数
- `mem_ptr` (int): 内存指针地址
- `abbreviated` (bool): 是否输出简略摘要

## 4. 返回值
各函数返回类型不同：
- 内存统计：Dict[str, Any] 或 int
- 内存信息：Tuple[int, int] (空闲内存, 总内存)
- 操作函数：None 或内存指针
- 摘要信息：str 格式文本

## 5. 文档要点
- 需要 CUDA 设备支持，部分函数依赖 `is_initialized()`
- 内存统计分为：allocated、reserved、active、inactive 等类别
- 支持大池(≥1MB)和小池(<1MB)分别统计
- 包含当前值、峰值、累计分配、累计释放四种指标

## 6. 源码摘要
- 核心函数调用底层 C++ API：`torch._C._cuda_*`
- 内存统计通过 `memory_stats_as_nested_dict` 获取嵌套字典
- `memory_summary` 格式化输出人类可读摘要
- `list_gpu_processes` 依赖 pynvml 库查询进程信息
- 副作用：修改 GPU 内存分配器状态、释放缓存

## 7. 示例与用法（如有）
模块级示例：
```python
import torch
# 查询当前内存使用
allocated = torch.cuda.memory_allocated()
# 清空缓存
torch.cuda.empty_cache()
# 获取详细统计
stats = torch.cuda.memory_stats()
```

## 8. 风险与空白
- **多实体情况**：模块包含 20+ 个函数，需分别测试
- **设备依赖**：需要实际 CUDA 设备才能测试完整功能
- **外部依赖**：`list_gpu_processes` 需要 pynvml 库
- **类型注解不完整**：部分函数缺少返回类型注解
- **边界条件**：内存分配失败、设备不存在等异常处理
- **并发安全**：多线程/多进程环境下的内存操作
- **平台差异**：不同 CUDA 版本可能行为不同