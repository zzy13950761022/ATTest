# torch._lowrank 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 验证低秩矩阵近似算法正确性：get_approximate_basis、svd_lowrank、pca_lowrank
  - 确保随机投影和子空间迭代算法符合 Halko et al, 2009 算法 4.4 和 5.1
  - 验证稀疏和稠密张量处理能力
  - 确认中心化参数对 PCA 结果的影响
- 不在范围内的内容
  - 完整 SVD 算法（torch.linalg.svd）
  - 非低秩矩阵的极端性能测试
  - 与其他库（如 scipy）的基准比较

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - A: Tensor, 形状 (*, m, n), 无默认值
  - q: int, 子空间维度, 默认值: get_approximate_basis 无默认, svd_lowrank=6, pca_lowrank=min(6, m, n)
  - niter: int, 子空间迭代次数, 默认值: 2
  - M: Tensor, 均值张量, 形状 (*, 1, n), 默认值: None
  - center: bool, 是否中心化, 默认值: True
- 有效取值范围/维度/设备要求
  - q 范围: 0 ≤ q ≤ min(m, n)
  - niter ≥ 0
  - 支持 CPU 和 GPU 设备
  - 支持 float32/float64 数据类型
  - pca_lowrank 支持稀疏张量
- 必需与可选组合
  - get_approximate_basis: A, q 必需; niter, M 可选
  - svd_lowrank: A 必需; q, niter, M 可选
  - pca_lowrank: A 必需; q, center, niter 可选
- 随机性/全局状态要求
  - 需要可重复结果时重置伪随机数生成器种子
  - 随机投影算法依赖 torch 随机状态

## 3. 输出与判定
- 期望返回结构及关键字段
  - get_approximate_basis: 正交基张量 Q, 形状 (*, m, q)
  - svd_lowrank: 元组 (U, S, V), U 形状 (*, m, q), S 形状 (*, q), V 形状 (*, n, q)
  - pca_lowrank: 元组 (U, S, V), 同 svd_lowrank
- 容差/误差界（如浮点）
  - 正交性验证: Q^T Q ≈ I (容差 1e-6)
  - 重构误差: A ≈ U diag(S) V^T (相对误差 ≤ 1e-4)
  - 奇异值排序: S 降序排列
- 状态变化或副作用检查点
  - 验证 torch 随机状态是否受影响
  - 检查输入张量是否被修改

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - q > min(m, n): 引发 ValueError
  - q < 0: 引发 ValueError
  - niter < 0: 引发 ValueError
  - 非 Tensor 输入: 引发 TypeError
  - M 形状不匹配: 引发 RuntimeError
  - 稀疏张量用于不支持函数: 引发 TypeError
- 边界值（空、None、0 长度、极端形状/数值）
  - q = 0: 返回空张量或零维度结果
  - q = min(m, n): 完整秩近似
  - niter = 0: 无子空间迭代
  - 极端形状: (1, 1), (1000, 10), (10, 1000)
  - 极端数值: 全零矩阵, 单位矩阵, 病态矩阵

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - 依赖 torch.linalg.qr 和 torch.linalg.svd
  - 需要 CUDA 环境进行 GPU 测试
  - 依赖 torch 随机数生成器
- 需要 mock/monkeypatch 的部分
  - torch.linalg.qr 和 torch.linalg.svd 的异常路径
  - 随机数生成器状态控制
  - 稀疏张量格式转换

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. 验证 q 参数边界条件 (0 ≤ q ≤ min(m, n))
  2. 测试稀疏和稠密张量输入的正确性
  3. 验证中心化参数对 pca_lowrank 结果的影响
  4. 检查不同 dtype (float32/float64) 和设备 (CPU/GPU) 的兼容性
  5. 确认随机算法可重复性（通过种子控制）
- 可选路径（中/低优先级合并为一组列表）
  - 极端形状矩阵测试（超大 m 或 n）
  - 不同 niter 值对精度的影响
  - M 参数对结果的影响验证
  - 病态矩阵的数值稳定性
  - 批量处理 (*, m, n) 形状的验证
  - 与完整 SVD 结果的对比分析
- 已知风险/缺失信息（仅列条目，不展开）
  - 随机算法结果可能轻微变化
  - 稠密矩阵性能警告（比完整 SVD 慢 10 倍）
  - 缺少具体数值示例参考
  - 稀疏张量仅 pca_lowrank 支持
  - q 选择准则验证（k ≤ q ≤ min(2*k, m, n)）