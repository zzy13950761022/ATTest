# tensorflow.python.saved_model.save 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（针对文件I/O和内部函数）
- 随机性处理：固定随机种子/控制RNG（不适用，函数无随机性）
- 文件系统隔离：使用临时目录，测试后清理

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04, CASE_05
- DEFERRED_SET: 无（首轮全覆盖）
- 测试文件路径：tests/test_tensorflow_python_saved_model_save.py（单文件）
- 断言分级策略：首轮使用weak断言，最终轮启用strong断言
- 预算策略：
  - size: S（小型测试）
  - max_lines: 75-90行
  - max_params: 6-7个参数
  - is_parametrized: false（首轮不参数化）
  - requires_mock: 大部分用例需要mock

## 3. 数据与边界
- 正常数据集：tf.Module对象、带@tf.function方法、变量、签名
- 随机生成策略：不适用（确定性测试）
- 边界值：
  - 空tf.Module对象
  - 空导出目录字符串
  - None签名参数
  - 空签名字典
- 极端形状：不适用（模型结构非张量形状）
- 负例与异常场景：
  - 非Trackable对象输入
  - 无效导出目录（权限不足）
  - 在@tf.function内部调用
  - 无效签名格式

## 4. 覆盖映射
| TC ID | 需求覆盖 | 约束覆盖 |
|-------|----------|----------|
| TC-01 | 基本tf.Module对象保存 | obj必须继承Trackable |
| TC-02 | 带@tf.function方法的模型 | 自动搜索@tf.function方法 |
| TC-03 | 显式signatures参数传递 | signatures参数格式验证 |
| TC-04 | 包含变量的可追踪对象 | 变量通过属性跟踪 |
| TC-05 | 无效Trackable对象异常 | 错误类型和消息验证 |

### 尚未覆盖的风险点
- TensorFlow 1.x图形模式支持
- 嵌套Trackable对象处理
- 资产文件处理机制
- 资源变量序列化细节
- 跨设备变量处理
- 性能影响和内存使用