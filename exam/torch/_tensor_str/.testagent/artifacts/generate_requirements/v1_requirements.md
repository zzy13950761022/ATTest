# torch._tensor_str 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试张量字符串格式化模块，验证 `set_printoptions()` 全局配置和 `_str()` 张量格式化功能，确保不同张量类型、形状、数值范围的正确字符串表示
- 不在范围内的内容：不测试 PyTorch 其他模块功能，不验证外部打印函数（如 `print()`），不覆盖 UI/交互式环境特殊行为

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - `set_printoptions()`: precision(int/None), threshold(int/None), edgeitems(int/None), linewidth(int/None), profile(str/None), sci_mode(bool/None)
  - `_str()`: self(Tensor), tensor_contents(str/None)
- 有效取值范围/维度/设备要求：
  - precision: 非负整数，None 表示默认
  - threshold: 非负整数，0 表示不截断
  - edgeitems: 非负整数
  - linewidth: 正整数
  - profile: 'default', 'short', 'full' 或 None
  - sci_mode: True/False/None
  - 张量：支持 CPU/CUDA 设备，任意形状和维度
- 必需与可选组合：
  - `_str()` 必须传入 Tensor 对象
  - `set_printoptions()` 所有参数可选，可部分设置
- 随机性/全局状态要求：
  - `set_printoptions()` 修改全局打印状态，需测试状态隔离和恢复
  - 测试需考虑并发环境下的状态污染风险

## 3. 输出与判定
- 期望返回结构及关键字段：
  - `_str()` 返回字符串，包含张量值、形状、dtype、设备信息
  - 字符串格式需符合预设精度、截断规则
  - 特殊张量（稀疏、量化）需有相应标识
- 容差/误差界（如浮点）：
  - 浮点数显示精度需严格匹配 precision 设置
  - 科学计数法切换阈值需验证
  - 复数实部/虚部分别格式化
- 状态变化或副作用检查点：
  - `set_printoptions()` 调用后全局打印行为变化
  - 不同线程/进程间的状态隔离
  - 异常情况下的状态回滚

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 非 Tensor 对象传入 `_str()`
  - 无效 profile 值
  - 负数的 precision/threshold/edgeitems
  - 非整数的 linewidth
- 边界值（空、None、0 长度、极端形状/数值）：
  - 空张量（shape 包含 0）
  - 单元素张量
  - 超大张量（触发截断）
  - 极端数值（NaN, inf, -inf, 0）
  - 极端形状（高维、不规则形状）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - PyTorch 库依赖
  - CUDA 设备（可选，用于 GPU 张量测试）
  - 无网络/文件系统依赖
- 需要 mock/monkeypatch 的部分：
  - 全局打印状态隔离
  - 设备可用性检查
  - 内存限制模拟（大张量处理）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基本浮点张量格式化与精度控制
  2. 大张量截断显示与 threshold 参数验证
  3. 稀疏/量化张量特殊标识显示
  4. 全局打印选项设置与恢复
  5. 复数张量实部/虚部分别格式化
- 可选路径（中/低优先级合并为一组列表）：
  - 不同设备（CPU/CUDA）显示一致性
  - 极端形状张量（>4维）格式化
  - 命名张量与梯度信息显示
  - 科学计数法切换阈值测试
  - 多线程环境状态隔离
  - 元张量/函数式张量特殊处理
  - 不同 dtype 组合测试
- 已知风险/缺失信息（仅列条目，不展开）：
  - 缺少完整类型注解
  - 并发状态管理细节不明确
  - 内存使用边界未文档化
  - 特殊张量类型（如 MPS）支持情况
  - 性能边界条件（超大张量处理时间）