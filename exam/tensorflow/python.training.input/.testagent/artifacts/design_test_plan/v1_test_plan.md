# tensorflow.python.training.input 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（队列运行器、队列对象、警告）
- 随机性处理：固定随机种子/控制 RNG（seed 参数）
- 执行模式：仅支持 graph execution 模式
- 弃用处理：捕获并验证弃用警告

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基本批量处理）、CASE_02（随机打乱批量）、CASE_03（动态填充）
- **DEFERRED_SET**: CASE_04（稀疏张量）、CASE_05（文件输入生产者）
- **测试文件路径**: tests/test_tensorflow_python_training_input.py（单文件）
- **断言分级策略**: 首轮使用 weak 断言，最终轮启用 strong 断言
- **预算策略**: 
  - Size: S（80行以内）/M（100行以内）
  - max_params: 5-9个参数
  - 所有用例参数化设计

## 3. 数据与边界
- **正常数据集**: 简单张量列表、变长张量、稀疏张量、模拟文件列表
- **随机生成策略**: 固定 seed 保证可重现性
- **边界值**: 
  - batch_size=1（最小批量）
  - capacity=batch_size（最小容量）
  - min_after_dequeue=0（最小出队后数量）
  - 零维张量处理
- **极端形状**: 超大容量队列（性能测试）
- **空输入**: 空张量列表触发 ValueError
- **负例与异常场景**:
  - batch_size ≤ 0 触发 ValueError
  - capacity < batch_size 触发 ValueError
  - 非张量输入触发 TypeError
  - 形状不兼容触发 ValueError
  - 弃用警告必须触发

## 4. 覆盖映射
| TC ID | 对应需求/约束 | 优先级 | 覆盖功能点 |
|-------|--------------|--------|------------|
| TC-01 | 基本批量处理功能验证 | High | batch() 基础功能，队列运行器添加 |
| TC-02 | 随机打乱批量验证 | High | shuffle_batch()，随机种子控制 |
| TC-03 | 动态填充功能测试 | High | dynamic_pad=True，变长张量处理 |
| TC-04 | 稀疏张量正确处理 | High | 稀疏张量格式保持 |
| TC-05 | 文件输入生产者基础功能 | High | string_input_producer() 基础功能 |

**尚未覆盖的风险点**:
- 多队列批量处理（batch_join/shuffle_batch_join）
- 切片输入生产者（slice_input_producer）
- 多 epoch 处理验证
- 队列关闭和资源清理
- 多线程竞态条件
- 形状推断边界情况
- 大容量队列性能测试

## 5. Mock 策略
- **必需 Mock 目标**:
  - tensorflow.python.training.queue_runner.QueueRunner
  - tensorflow.python.ops.data_flow_ops.FIFOQueue
  - tensorflow.python.ops.data_flow_ops.RandomShuffleQueue
  - tensorflow.python.training.queue_runner.add_queue_runner
  - warnings.warn（弃用警告捕获）

- **Mock 验证点**:
  - 队列运行器正确创建和添加
  - 队列类型正确选择（FIFO/RandomShuffle）
  - 弃用警告正确触发
  - 局部变量初始化调用

## 6. 迭代策略
- **首轮（round1）**: 仅生成 SMOKE_SET（3个用例），使用 weak 断言
- **后续轮（roundN）**: 修复失败用例，从 DEFERRED_SET 提升用例
- **最终轮（final）**: 启用 strong 断言，可选覆盖率检查