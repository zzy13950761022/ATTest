# torch.utils.data.dataset 测试需求

## 1. 目标与范围
- 验证 Dataset 抽象基类及其子类的正确实现
- 测试 TensorDataset、ConcatDataset、Subset 等具体类的功能
- 验证 random_split 函数的分割逻辑和随机性控制
- 不在范围内的内容：多进程数据加载器（DataLoader）、自定义数据集实现细节

## 2. 输入与约束
- **Dataset 类**：子类必须实现 `__getitem__(index)` 方法
- **TensorDataset**：`*tensors` 参数要求所有张量第一维度大小相同
- **random_split**：
  - `dataset`：必须为 Dataset 实例
  - `lengths`：整数序列或比例序列（总和为1）
  - `generator`：可选随机数生成器，默认使用全局状态
- **ConcatDataset**：不支持 IterableDataset 拼接
- **Subset**：索引必须在原数据集范围内

## 3. 输出与判定
- **TensorDataset**：返回元组形式的张量切片，保持原始数据类型
- **random_split**：返回 Subset 对象列表，长度与 lengths 参数一致
- **ConcatDataset**：正确拼接多个数据集，支持负索引访问
- **Subset**：正确映射索引到原数据集
- 状态变化：random_split 不应修改原数据集

## 4. 错误与异常场景
- **TensorDataset**：张量第一维度不一致时抛出 RuntimeError
- **random_split**：
  - lengths 总和与数据集大小不匹配时抛出 ValueError
  - 负长度或比例时抛出 ValueError
- **Dataset 抽象类**：直接实例化时 `__getitem__` 抛出 NotImplementedError
- **边界值**：
  - 空 TensorDataset
  - 长度为0的 random_split
  - 负索引访问 ConcatDataset
  - 超出范围的 Subset 索引
- **类型错误**：非 Dataset 对象传入 random_split

## 5. 依赖与环境
- **外部依赖**：torch.randperm（随机排列生成）
- **需要 mock 的部分**：
  - `torch.randperm`：控制 random_split 的随机性
  - 全局随机状态：测试 generator 参数默认行为
  - `bisect.bisect_right`：验证 ConcatDataset 索引查找
- **设备依赖**：TensorDataset 支持 CPU/CUDA 张量

## 6. 覆盖与优先级
- **必测路径（高优先级）**：
  1. TensorDataset 基本切片功能与张量维度验证
  2. random_split 整数分割和比例分割的正确性
  3. ConcatDataset 多数据集拼接与索引映射
  4. Dataset 抽象类接口约束验证
  5. Subset 索引重映射功能

- **可选路径（中/低优先级）**：
  - IterableDataset 与 ConcatDataset 的兼容性
  - 大规模数据集（>10^6 样本）的性能测试
  - 混合精度张量在 TensorDataset 中的处理
  - 自定义 Dataset 子类的继承行为
  - 多线程环境下的数据集访问

- **已知风险/缺失信息**：
  - 类型注解不完整（`__getitem__` 返回类型）
  - 多进程场景下的线程安全性未明确说明
  - 不支持非整数索引的详细约束
  - generator 参数默认值依赖全局状态