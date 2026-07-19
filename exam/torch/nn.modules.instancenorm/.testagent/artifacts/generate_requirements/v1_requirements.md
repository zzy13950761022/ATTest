# torch.nn.modules.instancenorm 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证1D/2D/3D实例归一化层及其惰性版本的正确性，包括前向传播、参数初始化、设备兼容性
- 不在范围内的内容：底层F.instance_norm函数实现、_NormBase基类内部逻辑、训练循环集成

## 2. 输入与约束
- 参数列表：
  - num_features (int): 输入通道数，必须为正整数
  - eps (float=1e-5): 数值稳定性小量，必须为正浮点数
  - momentum (float=0.1): 运行统计量更新动量，范围[0,1]
  - affine (bool=False): 是否学习缩放和偏移参数
  - track_running_stats (bool=False): 是否跟踪运行统计量
  - device/dtype: 可选，用于指定设备和数据类型

- 有效取值范围/维度要求：
  - InstanceNorm1d: 输入形状2D(无批次)或3D(批次,C,L)
  - InstanceNorm2d: 输入形状3D(无批次)或4D(批次,C,H,W)
  - InstanceNorm3d: 输入形状4D(无批次)或5D(批次,C,D,H,W)
  - 通道数必须匹配num_features

- 必需与可选组合：
  - num_features必需（惰性版本除外）
  - affine=False时，不学习γ和β参数
  - track_running_stats=False时，不维护运行统计量

- 随机性/全局状态要求：无全局状态依赖，随机性仅来自参数初始化

## 3. 输出与判定
- 期望返回结构：与输入形状相同的Tensor
- 容差/误差界：浮点计算误差在1e-5范围内
- 状态变化检查点：
  - affine=True时，γ和β参数应可训练
  - track_running_stats=True时，running_mean和running_var应更新
  - 无批次输入时自动添加批次维度处理

## 4. 错误与异常场景
- 非法输入异常：
  - num_features非正整数触发ValueError
  - eps非正浮点数触发ValueError
  - momentum超出[0,1]范围触发ValueError
  - 输入通道数与num_features不匹配触发RuntimeError
  - 输入维度不符合要求触发RuntimeError

- 边界值测试：
  - 单样本输入（批次大小=1）
  - 小通道数（num_features=1）
  - 极端形状（大尺寸/小尺寸）
  - 极端数值（接近0、极大值、NaN、Inf）
  - eps极小值（接近机器精度）

## 5. 依赖与环境
- 外部资源依赖：PyTorch库，可选CUDA设备
- 需要mock/monkeypatch部分：无外部网络/文件依赖
- 设备兼容性：CPU和GPU（如可用）均需测试
- 数据类型兼容性：float32、float64、bfloat16（如支持）

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. 基本前向传播：各维度实例归一化正确计算
  2. affine参数：True/False时γ和β参数行为验证
  3. track_running_stats：True/False时统计量更新逻辑
  4. 设备兼容性：CPU/GPU计算结果一致性
  5. 数据类型：float32/float64精度差异验证

- 可选路径（中/低优先级）：
  - 惰性版本自动推断num_features
  - 动量参数不同值的影响
  - 无批次输入自动处理
  - 梯度计算正确性
  - 序列化/反序列化（state_dict）
  - 混合精度训练兼容性
  - 极端形状和数值稳定性

- 已知风险/缺失信息：
  - 文档字符串不完整，部分约束需源码验证
  - 抽象方法_check_input_dim和_get_no_batch_dim需子类实现
  - 版本兼容性处理逻辑（_load_from_state_dict）
  - 标准差计算使用有偏估计器的影响