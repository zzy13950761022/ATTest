# torch.ao.quantization.fuse_modules 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证模块融合功能，支持 conv-bn、conv-bn-relu、conv-relu、linear-relu、bn-relu 序列融合，正确替换模块结构
- 不在范围内的内容：自定义融合函数内部逻辑、融合后模型量化效果、训练模式下的行为

## 2. 输入与约束
- 参数列表：
  - model: PyTorch Module，包含待融合模块
  - modules_to_fuse: list，字符串列表（单组）或列表的列表（多组）
  - inplace: bool，默认 False
  - fuser_func: callable，默认 fuse_known_modules
  - fuse_custom_config_dict: dict，默认 None
- 有效取值范围/维度/设备要求：模型需处于 eval() 模式，模块名称需存在且可访问
- 必需与可选组合：model 和 modules_to_fuse 必需，其余可选
- 随机性/全局状态要求：无随机性，不依赖全局状态

## 3. 输出与判定
- 期望返回结构及关键字段：PyTorch Module，包含融合模块
- 容差/误差界：融合前后模型输出误差 < 1e-6（浮点误差）
- 状态变化或副作用检查点：inplace=False 时返回新副本，融合后前向钩子丢失，模块结构正确替换

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常：非 Module 类型 model，非 list 类型 modules_to_fuse，不存在的模块名称
- 边界值：空列表，None 输入，重复模块名称，不支持融合的模块序列
- 极端形状/数值：大模型深度，嵌套子模块，复杂模块结构

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：PyTorch 库，CPU/GPU 设备
- 需要 mock/monkeypatch 的部分：fuser_func 自定义融合函数，fuse_custom_config_dict 配置处理

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. 单组模块融合（conv-bn-relu 序列）
  2. 多组模块融合（列表的列表）
  3. inplace=True/False 行为验证
  4. 不支持序列的模块保持不变
  5. 嵌套子模块融合
- 可选路径（中/低优先级）：
  - 自定义 fuser_func 调用
  - fuse_custom_config_dict 配置应用
  - 不同设备（CPU/GPU）兼容性
  - 大模型性能基准
  - 融合后模型序列化/反序列化
- 已知风险/缺失信息：
  - 未明确支持的 Conv/Linear/BatchNorm 具体类型
  - 自定义融合配置的具体格式限制
  - 融合对模型训练状态的影响
  - 融合后模块命名具体规则