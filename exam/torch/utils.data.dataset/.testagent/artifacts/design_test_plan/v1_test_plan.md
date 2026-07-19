# torch.utils.data.dataset 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG
- 测试范围：Dataset 抽象基类、TensorDataset、ConcatDataset、Subset、random_split

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03
- **DEFERRED_SET**: CASE_04, CASE_05
- **测试文件路径**: tests/test_torch_utils_data_dataset.py（单文件）
- **断言分级策略**: 首轮使用 weak 断言，最终轮启用 strong 断言
- **预算策略**: 
  - Size: S/M（小型/中型用例）
  - max_lines: 50-85 行
  - max_params: 2-5 个参数

## 3. 数据与边界
- **正常数据集**: 随机生成张量，形状多样（图像/标签对）
- **边界值**: 空数据集（0样本）、单样本数据集、大尺寸张量
- **极端形状**: 高维张量（4D）、标量张量、不同数据类型
- **空输入**: 空 TensorDataset、长度为0的 random_split
- **负例场景**: 
  - 张量维度不匹配
  - random_split 长度超限
  - 负索引越界
  - 非 Dataset 对象输入
  - 比例总和不为1

## 4. 覆盖映射
| TC ID | 对应需求/约束 | 覆盖风险点 |
|-------|--------------|-----------|
| TC-01 | TensorDataset 基本功能 | 张量切片正确性、数据类型保持 |
| TC-02 | TensorDataset 维度验证 | 异常处理、错误信息格式 |
| TC-03 | random_split 整数分割 | 随机性控制、样本分配完整性 |
| TC-04 | ConcatDataset 拼接 | 索引映射、二分查找逻辑 |
| TC-05 | Dataset 抽象类 | 接口约束、NotImplementedError |

**尚未覆盖的关键风险点**：
- 多进程环境下的线程安全性
- IterableDataset 与 ConcatDataset 兼容性
- 大规模数据集性能问题
- 混合精度张量处理
- 类型注解完整性验证

## 5. 迭代策略
- **首轮 (round1)**: 仅生成 SMOKE_SET（3个核心用例），使用 weak 断言
- **后续轮 (roundN)**: 修复失败用例，从 deferred_set 提升用例（每次最多3个）
- **最终轮 (final)**: 启用 strong 断言，可选覆盖率检查

## 6. Mock 策略
- **需要 mock 的目标**:
  - `torch.randperm`: 控制 random_split 随机性
  - `bisect.bisect_right`: 验证 ConcatDataset 索引查找
- **Mock 约束**: 与 requirements.md 中的 mock 要求一致，确保测试可重复性