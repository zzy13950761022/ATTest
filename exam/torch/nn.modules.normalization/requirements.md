# torch.nn.modules.normalization 测试需求

## 1. 目标与范围
- 测试四种归一化层：LocalResponseNorm、CrossMapLRN2d、LayerNorm、GroupNorm
- 验证前向传播正确性、参数有效性检查、设备/数据类型支持
- 不在范围：反向传播梯度计算、优化器集成、自定义初始化

## 2. 输入与约束
- **LocalResponseNorm**: size(int), alpha(float=1e-4), beta(float=0.75), k(float=1)
- **LayerNorm**: normalized_shape(int/list/torch.Size), eps(float=1e-5), elementwise_affine(bool=True), device/dtype可选
- **GroupNorm**: num_groups(int), num_channels(int), eps(float=1e-5), affine(bool=True), device/dtype可选
- **形状约束**: GroupNorm要求num_channels能被num_groups整除
- **设备要求**: 支持CPU/CUDA，数据类型支持float32/float64
- **随机性**: 无全局状态依赖，仅依赖输入数据统计量

## 3. 输出与判定
- 所有类forward方法返回Tensor，形状与输入相同
- 浮点容差：相对误差1e-5，绝对误差1e-7
- 状态变化：无副作用，训练/评估模式行为一致
- 仿射参数：当affine=True时，scale/bias参数应存在且可训练

## 4. 错误与异常场景
- GroupNorm：num_channels不能被num_groups整除时抛出ValueError
- 非法输入：非Tensor输入、维度不匹配、无效数据类型
- 边界值：size=0或负数、eps=0或负数、极端alpha/beta值
- 形状异常：normalized_shape与输入最后D维不匹配
- 设备不匹配：参数与输入张量设备不一致

## 5. 依赖与环境
- 外部依赖：torch、torch.nn.functional
- 设备依赖：需要CUDA环境测试GPU支持
- Mock需求：无需外部资源，纯计算模块
- 环境要求：Python 3.8+，PyTorch 1.9+

## 6. 覆盖与优先级
- **必测路径（高优先级）**：
  1. GroupNorm整除性检查异常处理
  2. LayerNorm不同normalized_shape形状支持
  3. LocalResponseNorm跨通道归一化正确性
  4. 设备/数据类型参数化测试
  5. affine/elementwise_affine参数开关测试

- **可选路径（中/低优先级）**：
  - CrossMapLRN2d与LocalResponseNorm差异对比
  - 极端输入值（极大/极小数值）稳定性
  - 批量大小=1边界情况
  - 不同维度输入（2D/3D/4D）支持
  - 训练/评估模式一致性验证

- **已知风险/缺失信息**：
  - CrossMapLRN2d文档字符串缺失
  - 设备/数据类型参数具体约束未详细说明
  - 缺少对CrossMapLRN2d与LocalResponseNorm差异的明确说明
  - 标准差计算使用有偏估计器的具体影响