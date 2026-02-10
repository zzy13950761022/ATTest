# tensorflow.lite.python.interpreter 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用mock模拟文件系统、模型加载和委托功能
- 随机性处理：固定随机种子确保可重复性
- 测试数据：使用简单加法模型作为基础测试模型

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07, CASE_08
- **group列表**: G1（核心Interpreter类功能）, G2（高级功能与异常处理）
- **active_group_order**: G1, G2
- **断言分级策略**: 首轮使用weak断言，最终轮启用strong断言
- **预算策略**: 
  - size: S/M（小型/中型）
  - max_lines: 65-90行
  - max_params: 4-6个参数
  - 首轮只生成4个核心用例

## 3. 数据与边界
- **正常数据集**: 简单加法模型（2x2 float32矩阵）
- **随机生成策略**: 固定种子生成随机输入数据
- **边界值**: 
  - 空模型文件（0字节）
  - 零长度输入张量（shape包含0）
  - num_threads边界值（-1, 0, 1, 4）
  - 超大shape测试内存限制
- **负例与异常场景**:
  - model_path和model_content同时为None
  - 无效模型文件路径
  - 损坏的模型二进制内容
  - 张量索引越界访问
  - 未分配张量直接调用invoke
  - 输入数据shape/dtype不匹配

## 4. 覆盖映射
- **TC-01**: 模型加载与张量分配 → 需求1,2,3
- **TC-02**: 完整推理流程 → 需求1,2,3,4
- **TC-03**: 签名运行器基本功能 → 需求1,5
- **TC-04**: 模型内容加载 → 需求1,2

**尚未覆盖的风险点**:
- 委托功能平台限制（仅CPython）
- 内存泄漏检测
- 并发访问测试
- 中间张量访问的未定义行为
- 量化模型scale/zero_point验证