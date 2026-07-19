# tensorflow.python.ops.candidate_sampling_ops 测试计划

## 1. 测试策略
- **单元测试框架**：pytest
- **隔离策略**：使用mock隔离底层C++操作和随机数生成器，通过monkeypatch控制随机种子
- **随机性处理**：固定随机种子确保测试可重现，通过seed参数控制随机数生成
- **依赖管理**：mock所有gen_candidate_sampling_ops底层操作，避免依赖TensorFlow C++实现

## 2. 生成规格摘要（来自test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03（首轮生成）
- **DEFERRED_SET**: CASE_04, CASE_05（后续迭代）
- **单文件路径**: tests/test_tensorflow_python_ops_candidate_sampling_ops.py
- **断言分级策略**: 首轮使用weak断言（形状、类型、值范围），后续启用strong断言（分布正确性、精度验证）
- **预算策略**: 每个用例size=S，max_lines≤80，max_params≤6，优先参数化测试

## 3. 数据与边界
- **正常数据集**: 使用小规模张量（batch_size≤4，num_true≤3，num_sampled≤8，range_max≤20）
- **随机生成策略**: 固定种子生成可重现测试数据，值域在[0, range_max-1]内
- **边界值**:
  - range_max=1（最小类别数）
  - num_sampled=1（最小采样数）
  - batch_size=1（最小批量）
  - num_true=1（最小目标数）
  - unique=true且num_sampled=range_max（边界约束）
- **负例与异常场景**:
  - true_classes值超出范围
  - num_sampled>range_max且unique=true
  - range_max≤0非法输入
  - num_sampled≤0非法输入
  - 类型不匹配异常

## 4. 覆盖映射
- **TC-01**: 验证uniform_candidate_sampler基础功能，覆盖均匀分布采样核心路径
- **TC-02**: 验证unique=true约束，覆盖num_sampled≤range_max边界条件
- **TC-03**: 验证compute_accidental_hits基础功能，覆盖意外命中计算
- **TC-04**: 验证log_uniform_candidate_sampler分布特性，覆盖对数均匀分布
- **TC-05**: 验证all_candidate_sampler全类别采样，覆盖测试用采样器

- **尚未覆盖的风险点**:
  - learned_unigram_candidate_sampler学习分布
  - fixed_unigram_candidate_sampler固定分布
  - 大规模range_max性能问题
  - GPU/TPU设备差异
  - 动态图模式行为差异