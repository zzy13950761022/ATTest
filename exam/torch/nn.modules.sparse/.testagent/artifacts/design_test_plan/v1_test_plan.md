# torch.nn.modules.sparse 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：每个测试用例独立实例化模块，使用 pytest fixtures 管理资源
- 随机性处理：固定随机种子（torch.manual_seed），确保测试可重现
- 设备支持：优先 CPU，CUDA 作为参数扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04（4个核心用例）
- **DEFERRED_SET**: CASE_05, CASE_06（2个中级用例）
- **group 列表**:
  - G1: Embedding 核心功能（CASE_01, CASE_02, CASE_05, CASE_06）
  - G2: EmbeddingBag 聚合功能（CASE_03, CASE_04）
- **active_group_order**: G1, G2（优先测试 Embedding，再测 EmbeddingBag）
- **断言分级策略**: 首轮仅使用 weak 断言（形状、类型、有限性、基础属性）
- **预算策略**: 
  - S 大小用例：max_lines=80, max_params=6
  - M 大小用例：max_lines=100, max_params=8
  - 所有用例均参数化，减少重复代码

## 3. 数据与边界
- **正常数据集**: 小规模参数（num_embeddings=10, embedding_dim=3），随机生成索引
- **边界值测试**:
  - 索引边界：0, num_embeddings-1, -1（负索引）
  - 形状边界：空输入、单元素、大规模参数
  - 数值边界：max_norm=0, 极小值, 极大值
  - 设备边界：CPU vs CUDA（参数扩展）
- **负例与异常场景**:
  - num_embeddings ≤ 0 触发 ValueError
  - embedding_dim ≤ 0 触发 ValueError
  - padding_idx 超出范围触发 ValueError
  - 输入张量非 IntTensor/LongTensor 触发 TypeError
  - 索引越界触发 RuntimeError
  - 不兼容的 per_sample_weights 和 mode 组合

## 4. 覆盖映射
| TC_ID | 需求/约束覆盖 | 优先级 | 状态 |
|-------|--------------|--------|------|
| TC-01 | Embedding 基础正向传播与形状验证 | High | SMOKE |
| TC-02 | padding_idx 特殊处理与梯度隔离 | High | SMOKE |
| TC-03 | EmbeddingBag sum 模式功能正确性 | High | SMOKE |
| TC-04 | EmbeddingBag mean 模式功能正确性 | High | SMOKE |
| TC-05 | max_norm 范数约束与权重修改 | Medium | DEFERRED |
| TC-06 | 稀疏梯度模式与密集梯度模式对比 | Medium | DEFERRED |

**尚未覆盖的风险点**:
- 空袋（empty bag）处理逻辑
- per_sample_weights 与各种 mode 组合
- include_last_offset 不同设置
- scale_grad_by_freq 梯度缩放效果
- from_pretrained 类方法
- 不同 PyTorch 版本行为差异

## 5. 迭代策略
- **首轮（round1）**: 仅生成 SMOKE_SET（4个用例），使用 weak 断言
- **后续轮次（roundN）**: 修复失败用例，逐步提升 DEFERRED 用例
- **最终轮次（final）**: 启用 strong 断言，可选覆盖率检查

## 6. 文件组织
- 主文件: `tests/test_torch_nn_modules_sparse.py`
- 分组文件:
  - G1: `tests/test_torch_nn_modules_sparse_embedding.py`
  - G2: `tests/test_torch_nn_modules_sparse_embeddingbag.py`
- 所有用例与 CASE_XX 一一对应，BLOCK_ID 稳定不变