# torch.nn.modules.fold 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 测试 `Fold` 类：将滑动局部块数组组合成张量（col2im操作）
  - 测试 `Unfold` 类：从输入张量中提取滑动局部块（im2col操作）
  - 验证两个类配合使用时的行为一致性
- 不在范围内的内容
  - 底层 functional 模块实现细节
  - 非图像类张量（如1D、5D+）
  - 梯度计算和反向传播

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - `Fold`: output_size(int/tuple), kernel_size(int/tuple), stride(int/tuple, default=1), padding(int/tuple, default=0), dilation(int/tuple, default=1)
  - `Unfold`: kernel_size(int/tuple), stride(int/tuple, default=1), padding(int/tuple, default=0), dilation(int/tuple, default=1)
- 有效取值范围/维度/设备要求
  - 仅支持3D（未批处理）或4D（批处理）图像类张量
  - 参数为int时自动复制到所有空间维度
  - 输入输出形状必须满足数学公式约束
- 必需与可选组合
  - `Fold.output_size` 和 `kernel_size` 必需
  - `Unfold.kernel_size` 必需
  - stride, padding, dilation 可选，默认值为1/0/1
- 随机性/全局状态要求
  - 无随机性
  - 无全局状态依赖

## 3. 输出与判定
- 期望返回结构及关键字段
  - `Fold`: 形状为 `(N, C, output_size[0], output_size[1], ...)` 的张量
  - `Unfold`: 形状为 `(N, C × ∏(kernel_size), L)` 的张量，L为总块数
- 容差/误差界（如浮点）
  - 浮点计算误差在机器精度范围内
  - Fold-Unfold组合测试中允许微小数值差异
- 状态变化或副作用检查点
  - 无I/O操作
  - 无全局状态修改
  - 模块实例状态保持不变

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 非int/tuple参数类型
  - 负值或零值参数
  - 不支持的张量维度（1D、2D、5D+）
  - 形状不满足数学公式约束
- 边界值（空、None、0长度、极端形状/数值）
  - kernel_size=0或负值
  - output_size小于kernel_size
  - 极大padding导致负输出尺寸
  - dilation值小于1

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - PyTorch库依赖
  - 支持CPU和CUDA设备
  - 无网络或文件系统依赖
- 需要mock/monkeypatch的部分
  - 底层F.fold()和F.unfold()函数（可选）
  - 无外部服务调用

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. Fold基本功能：int参数和tuple参数的正确处理
  2. Unfold基本功能：不同kernel_size和stride组合
  3. Fold-Unfold组合：验证数学一致性（允许重叠求和）
  4. 边界条件：最小有效输入尺寸和参数
  5. 错误处理：非法参数和形状的异常抛出
- 可选路径（中/低优先级合并为一组列表）
  - 不同dilation值组合测试
  - 极端padding值测试
  - 批量大小变化测试
  - 通道数变化测试
  - 不同dtype支持测试（float32, float64）
  - 设备兼容性测试（CPU, CUDA）
  - 内存使用和性能基准
- 已知风险/缺失信息（仅列条目，不展开）
  - 具体支持的dtype范围未明确
  - 错误消息格式和类型未详细说明
  - 超大张量内存处理边界
  - 非标准形状的公式约束验证
  - 参数类型注解`_size_any_t`的具体约束