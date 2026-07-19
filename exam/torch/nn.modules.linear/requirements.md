# torch.nn.modules.linear 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试 Linear、Bilinear、Identity、LazyLinear 类的正确性，验证 y = xA^T + b 线性变换，支持多种输入形状 (*, in_features) → (*, out_features)
- 不在范围内的内容：量化相关类 NonDynamicallyQuantizableLinear 的特殊用途，第三方扩展实现

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - Linear: in_features(int>0), out_features(int>0), bias(bool/True), device(None), dtype(None)
  - forward: input(Tensor, shape(*, in_features))
- 有效取值范围/维度/设备要求：in_features>0, out_features>0，支持 CPU/CUDA，兼容 float16/float32/float64
- 必需与可选组合：in_features 和 out_features 必需，bias 可选默认 True
- 随机性/全局状态要求：权重使用 kaiming_uniform 初始化，偏置使用均匀分布 U(-1/√fan_in, 1/√fan_in)

## 3. 输出与判定
- 期望返回结构及关键字段：Tensor 形状 (*, out_features)，除最后一维外与输入形状相同
- 容差/误差界（如浮点）：float32 使用 1e-5 相对误差，float16 使用 1e-3 相对误差
- 状态变化或副作用检查点：LazyLinear 首次 forward 后 in_features 被推断并固定，权重和偏置参数正确初始化

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：in_features≤0 或 out_features≤0 触发 ValueError，输入张量最后一维≠in_features 触发 RuntimeError，非数值类型输入触发 TypeError
- 边界值（空、None、0 长度、极端形状/数值）：in_features=1 最小维度，大维度（如 10000）内存检查，极端数值（inf, nan）传播行为，空批次维度 (*=0)

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：PyTorch 库，CUDA 设备（可选），无网络/文件依赖
- 需要 mock/monkeypatch 的部分：F.linear 函数调用验证，随机初始化种子控制，设备内存模拟

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. Linear 基础正向传播验证正确形状和数值
  2. bias=False 配置验证无偏置项
  3. LazyLinear 延迟初始化首次推断 in_features
  4. Bilinear 双输入变换 y = x₁ᵀAx₂ + b
  5. Identity 层恒等映射验证
- 可选路径（中/低优先级合并为一组列表）：
  - 不同 dtype（float16/32/64）精度验证
  - 不同设备（CPU/CUDA）一致性检查
  - 极端形状（超大/超小维度）内存和性能
  - 梯度计算和反向传播正确性
  - 序列化（state_dict）保存加载
  - 训练模式切换（train/eval）行为
- 已知风险/缺失信息（仅列条目，不展开）：
  - in_features=0 或 out_features=0 的边界行为未定义
  - TensorFloat32 和 ROCm float16 的特殊精度处理
  - 量化相关类的特殊用途未覆盖
  - 多线程环境下的并发安全性