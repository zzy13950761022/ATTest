# tensorflow.python.data.experimental.ops.parsing_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures 用于内部依赖
- 随机性处理：固定随机种子，控制数据集生成
- 测试级别：单元测试，验证函数行为而非底层实现

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_05（4个核心用例）
- **DEFERRED_SET**: CASE_04, CASE_06, CASE_07, CASE_08（4个延后用例）
- **分组策略**: 2个测试组（G1:核心功能，G2:特征类型与边界）
- **断言分级**: 首轮使用weak断言，最终轮启用strong断言
- **预算控制**: 
  - S级用例：max_lines≤70, max_params≤5
  - M级用例：max_lines≤90, max_params≤7
  - 所有用例优先使用weak断言

## 3. 数据与边界
- **正常数据集**: 随机生成序列化Example protos，包含多种特征类型
- **边界值**: 空数据集、单元素数据集、超大字符串、极端数值
- **形状边界**: 空形状、高维形状、不规则形状
- **数据类型**: float32/64, int32/64, string, bool
- **负例场景**: 
  - features=None或空字典
  - num_parallel_calls≤0
  - 非字符串向量输入数据集
  - 无效protobuf格式
  - 不匹配的特征定义

## 4. 覆盖映射
| TC ID | 需求覆盖 | 约束覆盖 | 优先级 |
|-------|----------|----------|--------|
| TC-01 | 基本功能验证 | FixedLenFeature支持 | High |
| TC-02 | 参数验证 | features不能为None/空 | High |
| TC-03 | 并行解析 | num_parallel_calls>1 | High |
| TC-04 | 边界验证 | num_parallel_calls验证 | Medium |
| TC-05 | 特征类型 | 多种特征类型支持 | High |
| TC-06 | 确定性控制 | deterministic参数行为 | Medium |
| TC-07 | 边界处理 | 空数据集处理 | Low |
| TC-08 | 输入验证 | 数据集格式验证 | Medium |

## 5. 风险与未覆盖点
- **关键风险**:
  - deterministic=None时依赖数据集选项（环境依赖）
  - SparseFeature和RaggedFeature需要额外映射步骤
  - 内部API依赖（_ParseOpParams等）
  - 并行解析可能引入非确定性行为
- **未覆盖点**:
  - 性能基准测试（需要strong断言）
  - 内存使用监控
  - 与其他数据集操作的组合使用
  - 大规模数据集压力测试

## 6. 迭代策略
- **首轮**: 仅生成SMOKE_SET用例，使用weak断言
- **后续轮**: 修复失败用例，从DEFERRED_SET提升优先级
- **最终轮**: 启用strong断言，可选覆盖率检查

## 7. 依赖与Mock
- **需要Mock**: 
  - gen_experimental_dataset_ops.parse_example_dataset_v2
  - parsing_ops._ParseOpParams内部类
  - 底层C++操作调用
- **无需Mock**:
  - tf.io.parse_example（作为oracle）
  - 标准TensorFlow数据类型操作