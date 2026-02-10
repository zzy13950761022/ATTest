# torch.onnx.utils 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures 控制外部依赖
- 随机性处理：固定随机种子，控制 torch.jit.trace 行为
- 状态管理：mock GLOBALS 全局状态，确保测试隔离

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（基本模型导出到文件）、CASE_02（模型导出到BytesIO缓冲区）、CASE_03（导出状态检测函数）
- **DEFERRED_SET**: CASE_04（ScriptModule模型导出）、CASE_05（动态轴配置导出）等6个用例
- **group 列表**: 
  - G1: 核心导出函数族（export, export_to_pretty_string）
  - G2: 辅助函数与状态管理（is_in_onnx_export, _decide_keep_init_as_input）
- **active_group_order**: ["G1", "G2"] - 优先测试核心导出功能
- **断言分级策略**: 首轮使用weak断言（文件存在性、无异常等），后续启用strong断言（模型结构保持、数值精度等）
- **预算策略**: 
  - size: S(60行)/M(100行)/L(120行)
  - max_params: 4-9个参数
  - 参数化测试优先，减少重复代码

## 3. 数据与边界
- **正常数据集**: 简单线性模型、卷积网络、多层感知机
- **随机生成策略**: 固定种子生成随机权重，确保可重现
- **边界值**: 
  - 空模型（仅forward方法）
  - 1x1x1极端形状张量
  - opset_version边界值7和16
  - 动态轴空配置与复杂配置
- **负例与异常场景**:
  - 无效模型对象
  - 不支持的数据类型
  - 文件权限错误
  - 内存不足模拟
  - opset_version超出范围

## 4. 覆盖映射
- **TC-01**: 覆盖需求中的三种模型类型基本导出、文件输出目标
- **TC-02**: 覆盖BytesIO输出目标、张量格式参数
- **TC-03**: 覆盖全局状态管理、导出状态检测
- **TC-04**: 覆盖ScriptModule类型、避免torch.jit.trace调用
- **TC-05**: 覆盖动态轴配置、字典格式支持

- **尚未覆盖的风险点**:
  - 复杂模型结构（嵌套、循环、条件）导出能力
  - 大模型导出的性能和时间消耗
  - 多线程环境下的并发安全性
  - 不同operator_export_type的行为差异
  - 训练模式与opset_version>=12的关联验证