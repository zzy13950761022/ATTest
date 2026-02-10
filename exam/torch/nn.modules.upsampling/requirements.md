# torch.nn.modules.upsampling 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 测试 Upsample、UpsamplingNearest2d、UpsamplingBilinear2d 三个类的正确性
  - 验证 1D/2D/3D 数据上采样功能
  - 检查不同插值模式：nearest、linear、bilinear、bicubic、trilinear
  - 验证 size 和 scale_factor 参数的正确处理
  - 测试 align_corners 参数对线性插值模式的影响
- 不在范围内的内容
  - 底层 F.interpolate() 函数的内部实现
  - 非 PyTorch 张量输入的处理
  - 自定义插值算法的实现

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - size: int 或元组，可选，默认 None
  - scale_factor: float 或元组，可选，默认 None  
  - mode: str，默认 'nearest'，可选值：'nearest'、'linear'、'bilinear'、'bicubic'、'trilinear'
  - align_corners: bool，可选，默认 None
  - recompute_scale_factor: bool，可选，默认 None
- 有效取值范围/维度/设备要求
  - 输入形状：3D (N,C,W)、4D (N,C,H,W)、5D (N,C,D,H,W)
  - 输出形状：对应维度按指定比例放大
  - 支持 CPU 和 CUDA 设备
  - 支持常见 dtype：float32、float64
- 必需与可选组合
  - size 和 scale_factor 不能同时指定，必须至少指定一个
  - align_corners 仅对线性插值模式有效
  - UpsamplingNearest2d 固定 mode='nearest'
  - UpsamplingBilinear2d 固定 mode='bilinear'，align_corners=True
- 随机性/全局状态要求
  - 无随机性要求
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - 类实例化返回 Module 对象
  - forward() 返回上采样后的 Tensor
  - 输出通道数与输入相同
  - 空间维度按指定比例放大
- 容差/误差界（如浮点）
  - 最近邻插值：精确匹配
  - 线性插值：浮点容差 1e-5
  - 双线性/三线性插值：浮点容差 1e-5
  - 双三次插值：浮点容差 1e-4
- 状态变化或副作用检查点
  - 无状态变化
  - 无副作用
  - 不修改输入张量

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - size 和 scale_factor 同时指定：ValueError
  - size 和 scale_factor 都未指定：ValueError
  - 无效 mode 值：ValueError
  - 不支持的输入维度：RuntimeError
  - 无效数据类型：RuntimeError
  - align_corners 用于非线性模式：警告
- 边界值（空、None、0 长度、极端形状/数值）
  - scale_factor=1.0：输出应与输入相同
  - scale_factor=0.0：应触发错误
  - scale_factor 负值：应触发错误
  - 空张量输入：应触发错误
  - 单元素张量：应正确处理
  - 极大尺寸输入：内存边界测试
  - 极小尺寸输入：正确性验证

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - PyTorch 库依赖
  - CUDA 设备（可选）
  - 无网络/文件依赖
- 需要 mock/monkeypatch 的部分
  - F.interpolate() 调用可 mock 以测试错误处理
  - 设备可用性检查
  - 内存分配失败场景

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. Upsample 类 size 和 scale_factor 互斥性验证
  2. 三种插值模式（nearest、bilinear、bicubic）基础功能测试
  3. align_corners 对线性插值的影响验证
  4. 输入维度 3D/4D/5D 的正确处理
  5. UpsamplingNearest2d 和 UpsamplingBilinear2d 子类功能
- 可选路径（中/低优先级合并为一组列表）
  - 不同 dtype（float16、float32、float64）支持
  - recompute_scale_factor 参数行为
  - 极端 scale_factor 值（如 0.5、2.0、10.0）
  - 批量处理不同尺寸输入
  - CUDA 与 CPU 结果一致性
  - 内存使用和性能基准
  - 梯度计算正确性（训练模式）
- 已知风险/缺失信息（仅列条目，不展开）
  - 内部类型 _size_any_t、_ratio_any_t 的具体定义
  - recompute_scale_factor 的详细行为说明
  - 特定 dtype 的精度限制
  - 内存不足时的错误处理机制
  - 多线程环境下的行为