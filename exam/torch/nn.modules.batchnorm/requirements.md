# torch.nn.modules.batchnorm 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试批量归一化模块的7个主要类（BatchNorm1d/2d/3d及其懒加载版本、SyncBatchNorm）的前向传播、训练/评估模式切换、统计量更新
- 不在范围内的内容：分布式环境下的SyncBatchNorm完整测试、与其他模块的集成测试、性能基准测试

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - num_features (int): 必需，特征数/通道数C
  - eps (float=1e-5): 数值稳定性分母
  - momentum (float=0.1或None): 运行统计量更新动量
  - affine (bool=True): 是否启用可学习参数
  - track_running_stats (bool=True): 是否跟踪运行统计量
- 有效取值范围/维度/设备要求：
  - BatchNorm1d: 输入形状(N, C)或(N, C, L)
  - BatchNorm2d: 输入形状(N, C, H, W)
  - BatchNorm3d: 输入形状(N, C, D, H, W)
  - SyncBatchNorm: 至少2D输入
  - num_features > 0
  - eps > 0
  - momentum ∈ [0, 1]或None
- 必需与可选组合：num_features必需，其他参数可选
- 随机性/全局状态要求：训练模式更新运行统计量，评估模式使用固定统计量

## 3. 输出与判定
- 期望返回结构及关键字段：输出张量形状与输入相同，保持设备与数据类型
- 容差/误差界（如浮点）：浮点误差在1e-5范围内，eps参数影响数值稳定性
- 状态变化或副作用检查点：
  - 训练模式：更新running_mean/running_var
  - 评估模式：使用running_mean/running_var（若track_running_stats=True）
  - affine=True时更新weight/bias参数

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - num_features ≤ 0触发ValueError
  - 输入维度不符合类要求触发RuntimeError
  - 输入类型非Tensor触发TypeError
  - eps ≤ 0触发ValueError
  - momentum超出[0,1]范围触发ValueError
- 边界值（空、None、0长度、极端形状/数值）：
  - 批量大小N=1（小批量统计量不稳定）
  - 极端eps值（如1e-10, 1.0）
  - momentum=None（累积移动平均）
  - 输入值全为0或极大/极小值

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：SyncBatchNorm需要分布式环境
- 需要mock/monkeypatch的部分：
  - 训练/评估模式切换
  - 随机数生成器（统计量初始化）
  - F.batch_norm函数调用（验证参数传递）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. BatchNorm1d/2d/3d基础前向传播
  2. 训练/评估模式切换与统计量更新
  3. affine=False和track_running_stats=False场景
  4. 输入形状边界验证
  5. 懒加载类的延迟初始化
- 可选路径（中/低优先级合并为一组列表）：
  - 不同设备（CPU/GPU）测试
  - 不同数据类型（float32/float64）测试
  - momentum=None的累积平均行为
  - 极端eps值对数值稳定性的影响
  - 批量大小N=1的特殊处理
  - 输入值全为0或常数的处理
- 已知风险/缺失信息（仅列条目，不展开）：
  - SyncBatchNorm分布式测试环境
  - 多GPU并行训练场景
  - 与自动混合精度（AMP）的兼容性
  - 梯度检查与反向传播验证
  - 序列化/反序列化后的状态恢复