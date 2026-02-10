# tensorflow.python.framework.test_util 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures 用于设备检测、分布式集群、文件系统
- 随机性处理：固定随机种子，控制 TensorFlow 图形和 eager 模式状态

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04, CASE_05
- DEFERRED_SET: CASE_06, CASE_07, CASE_08, CASE_09, CASE_10, CASE_11
- group 列表与 active_group_order: G1(测试基类与装饰器), G2(图形比较与断言函数), G3(设备管理与测试环境)
- 断言分级策略：首轮使用 weak 断言（基础功能验证），后续启用 strong 断言（完整行为验证）
- 预算策略：size S(65-75行)/M(85-90行)，max_params 4-6，优先参数化测试

## 3. 数据与边界
- 正常数据集：简单图形结构、基本张量运算、标准设备配置
- 随机生成策略：固定种子生成可重复测试数据
- 边界值：空 GraphDef、零节点图形、None 设备、最小集群配置
- 极端形状：大节点数图形、复杂属性结构
- 空输入：空测试方法、无参数装饰器应用
- 负例与异常场景：
  - 无效 GraphDef 格式
  - 类型不匹配的断言参数
  - 装饰器应用于非测试方法
  - 设备不可用时的降级行为
  - 端口冲突的集群创建

## 4. 覆盖映射
- TC-01 (CASE_01): 覆盖 TensorFlowTestCase 基类继承和基本断言方法
- TC-02 (CASE_02): 覆盖 run_in_graph_and_eager_modes 装饰器模式切换
- TC-03 (CASE_03): 覆盖 assert_equal_graph_def 图形比较核心功能
- TC-04 (CASE_04): 覆盖 gpu_device_name 设备检测基础行为
- TC-05 (CASE_05): 覆盖 create_local_cluster 分布式测试环境创建

- 尚未覆盖的风险点：
  - 100+ 函数 API 的完整覆盖
  - GPU 硬件依赖的实际环境测试
  - 复杂状态管理场景的隔离性
  - 版本兼容性装饰器的跨版本测试
  - 临时文件管理的并发安全性