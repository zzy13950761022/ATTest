# tensorflow.python.ops.gen_decode_proto_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures 用于依赖隔离
- 随机性处理：固定随机种子确保测试可重复性
- 执行模式：支持 eager 和 graph 模式测试

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03（首轮生成）
- **DEFERRED_SET**: CASE_04, CASE_05（后续迭代）
- **测试文件路径**: tests/test_tensorflow_python_ops_gen_decode_proto_ops.py（单文件）
- **断言分级策略**: 首轮使用 weak 断言，最终轮启用 strong 断言
- **预算策略**: 
  - 每个用例最大 80-85 行代码
  - 最多 6 个参数
  - 用例规模为 S（小型）

## 3. 数据与边界
- **正常数据集**: 使用 tensorflow.Summary.Value 等标准 proto 消息
- **随机生成策略**: 固定种子生成测试 proto 数据
- **边界值**:
  - 空 field_names 列表
  - 空 bytes 张量
  - 超大 batch_size（内存边界）
  - 嵌套消息字段
- **极端形状**: 零尺寸维度防止机制
- **空输入**: 空张量和空字段列表
- **负例与异常场景**:
  - 无效 proto 数据
  - field_names 与 output_types 长度不匹配
  - 无效 message_type 名称
  - 不支持的 output_types 类型
  - 无效 descriptor_source 格式
  - 不支持的 message_format 值

## 4. 覆盖映射
| TC ID | 需求/约束覆盖 | 优先级 |
|-------|--------------|--------|
| TC-01 | 基本功能验证、有效 proto 解码 | High |
| TC-02 | 参数验证异常、长度不匹配 | High |
| TC-03 | 格式支持、二进制和文本格式 | High |
| TC-04 | 边界情况、空输入处理 | High |
| TC-05 | 数据类型支持、所有 TF 类型 | High |

### 尚未覆盖的风险点
- sanitize 参数的具体清理行为
- 子消息字段只能转换为 DT_STRING 的限制
- 文件路径 descriptor_source 的错误处理细节
- 并发调用时的线程安全性
- 超大 batch_size 的性能和内存使用

## 5. 迭代策略
1. **首轮 (round1)**: 生成 SMOKE_SET 用例，使用 weak 断言
2. **后续轮 (roundN)**: 修复失败用例，提升 DEFERRED_SET 用例
3. **最终轮 (final)**: 启用 strong 断言，可选覆盖率检查

## 6. Mock 策略
- CASE_05 需要 mock TensorFlow 内部函数
- mock_targets 包括 convert_to_tensor、_apply_op_helper、executing_eagerly
- 确保与 requirements.md 中的 mock 约束一致