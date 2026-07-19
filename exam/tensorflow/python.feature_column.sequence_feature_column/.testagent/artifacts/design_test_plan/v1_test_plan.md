# tensorflow.python.feature_column.sequence_feature_column 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock文件读取操作，使用fixtures管理TensorFlow会话
- 随机性处理：固定随机种子确保可重现性
- 设备策略：优先CPU测试，GPU作为可选扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_05, CASE_07（4个核心用例）
- **DEFERRED_SET**: CASE_03, CASE_04, CASE_06, CASE_08, CASE_09（5个延期用例）
- **group列表**: G1（核心创建函数）、G2（上下文拼接）、G3（参数验证）
- **active_group_order**: G1 → G2 → G3（按功能复杂度排序）
- **断言分级策略**: 首轮仅使用weak断言（类型检查、形状验证、异常捕获）
- **预算策略**: 每个用例size=S/M，max_lines≤80，max_params≤6

## 3. 数据与边界
- **正常数据集**: 标准形状张量（[batch, seq_len, features]），小规模词汇表
- **边界值**: 空字符串key，num_buckets=1，shape包含0，极大哈希桶
- **极端形状**: 单批次长序列，多批次短序列，高维特征
- **空输入**: 空词汇列表，空文件路径，None参数
- **负例场景**: 
  - num_buckets < 1触发ValueError
  - hash_bucket_size ≤ 1参数无效
  - default_value与num_oov_buckets互斥
  - 不支持dtype类型错误

## 4. 覆盖映射
| TC_ID | 覆盖需求 | 优先级 | 断言级别 |
|-------|----------|--------|----------|
| TC-01 | 序列分类特征列基本创建 | High | weak |
| TC-02 | 序列数值特征列创建 | High | weak |
| TC-03 | 词汇表文件特征列创建 | Medium | weak |
| TC-04 | 词汇列表特征列创建 | Medium | weak |
| TC-05 | 上下文输入拼接基本功能 | High | weak |
| TC-06 | 不同形状上下文拼接 | Medium | weak |
| TC-07 | 参数边界值异常验证 | High | weak |

**尚未覆盖的风险点**:
- API处于开发中，可能频繁变更
- 依赖TensorFlow内部不稳定API
- 多版本兼容性未明确
- 性能基准数据缺失
- 与embedding_column集成测试（依赖外部模块）

## 5. 迭代策略
- **首轮**: 仅生成SMOKE_SET用例，使用weak断言
- **后续轮次**: 修复失败用例，逐步启用DEFERRED_SET
- **最终轮**: 启用strong断言，可选覆盖率检查

## 6. 模块拆分
- **G1**: 5个核心创建函数（identity/hash/vocab/numeric）
- **G2**: 1个上下文拼接函数（concatenate_context_input）
- **G3**: 所有函数的参数验证和异常处理