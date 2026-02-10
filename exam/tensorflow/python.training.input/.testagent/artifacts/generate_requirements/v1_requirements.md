# tensorflow.python.training.input 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证基于队列的批量数据处理功能，包括文件匹配、批量处理、随机打乱等操作的正确性
- 不在范围内的内容：`tf.data` API 替代方案、eager execution 模式、分布式训练场景

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - `tensors`/`tensor_list`: Tensor 列表或字典，无默认值
  - `batch_size`: 整数 > 0，无默认值
  - `capacity`: 整数 ≥ batch_size，默认值：32
  - `num_epochs`: 整数 ≥ 1，默认值：None（无限）
  - `shuffle`: 布尔值，默认值：False
  - `seed`: 整数，默认值：None
  - `enqueue_many`: 布尔值，默认值：False
  - `dynamic_pad`: 布尔值，默认值：False
  - `allow_smaller_final_batch`: 布尔值，默认值：False
  - `min_after_dequeue`: 整数 ≥ 0，默认值：capacity // 4

- 有效取值范围/维度/设备要求：
  - batch_size 必须为正整数
  - capacity 必须 ≥ batch_size
  - min_after_dequeue 必须 ≤ capacity
  - 张量形状必须一致或兼容动态填充
  - 仅支持 graph execution 模式

- 必需与可选组合：
  - `tensors` 和 `batch_size` 为必需参数
  - `shuffle=True` 时建议设置 `seed` 保证可重现性
  - `dynamic_pad=True` 时张量形状可以不同

- 随机性/全局状态要求：
  - 使用 `seed` 参数控制随机性
  - 需要手动调用 `local_variables_initializer()`
  - 队列运行器添加到全局 `QUEUE_RUNNER` 集合

## 3. 输出与判定
- 期望返回结构及关键字段：
  - 批量函数返回 Tensor 列表或字典，形状为 [batch_size, ...]
  - 输入生产者函数返回 Queue 对象
  - 稀疏张量保持稀疏格式

- 容差/误差界（如浮点）：
  - 浮点数值误差在 1e-6 范围内
  - 批量顺序在 shuffle=False 时保持原序
  - 批量大小误差为 0（除非 allow_smaller_final_batch=True）

- 状态变化或副作用检查点：
  - 验证队列运行器正确添加到 `QUEUE_RUNNER`
  - 检查局部变量正确初始化
  - 确认无资源泄漏（队列关闭）

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - batch_size ≤ 0 触发 ValueError
  - capacity < batch_size 触发 ValueError
  - 非张量输入触发 TypeError
  - 形状不兼容触发 ValueError
  - 弃用警告必须触发

- 边界值（空、None、0 长度、极端形状/数值）：
  - 空张量列表触发 ValueError
  - None 输入触发 TypeError
  - 极端 batch_size（如 1, 1000000）验证性能
  - 零维张量处理
  - 超大容量队列内存检查

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - 文件系统访问（string_input_producer）
  - 线程池资源
  - TensorFlow graph execution 环境

- 需要 mock/monkeypatch 的部分：
  - `tensorflow.python.training.queue_runner.QueueRunner`
  - `tensorflow.python.ops.data_flow_ops.FIFOQueue`
  - `tensorflow.python.ops.data_flow_ops.RandomShuffleQueue`
  - `tensorflow.python.training.queue_runner.add_queue_runner`
  - `warnings.warn`（弃用警告捕获）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. 基本批量处理功能验证（shuffle=False）
  2. 随机打乱批量验证（shuffle=True，固定 seed）
  3. 动态填充功能测试（dynamic_pad=True）
  4. 稀疏张量正确处理
  5. 文件输入生产者基础功能

- 可选路径（中/低优先级合并为一组列表）：
  - 多队列批量处理（batch_join）
  - 随机打乱多队列（shuffle_batch_join）
  - 切片输入生产者（slice_input_producer）
  - 允许更小最终批次（allow_smaller_final_batch=True）
  - 多 epoch 处理验证
  - 大容量队列性能测试
  - 形状推断失败场景
  - 队列关闭和资源清理

- 已知风险/缺失信息（仅列条目，不展开）：
  - 所有函数已弃用
  - 不支持 eager execution
  - 需要手动变量初始化
  - 队列机制可能死锁
  - 多线程竞态条件
  - 形状推断边界情况
  - 稀疏张量复杂处理