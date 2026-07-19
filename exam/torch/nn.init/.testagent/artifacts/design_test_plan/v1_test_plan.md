# torch.nn.init 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：固定随机种子，使用torch.no_grad()上下文
- 随机性处理：控制随机种子，统计验证分布参数
- 设备隔离：优先CPU测试，可选GPU扩展

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_05, CASE_06, CASE_09（5个核心用例）
- **DEFERRED_SET**: CASE_03, CASE_04, CASE_07, CASE_08, CASE_10, CASE_11（6个延期用例）
- **group列表**: 
  - G1: 基础分布初始化（uniform_, normal_, constant_等）
  - G2: 自适应初始化策略（xavier_, kaiming_等）
  - G3: 特殊初始化函数（eye_, dirac_, sparse_等）
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮使用weak断言（shape/dtype/finite等），最终轮启用strong断言（分布验证/统计特性）
- **预算策略**: 
  - S级用例：max_lines≤80, max_params≤6
  - M级用例：max_lines≤85, max_params≤6
  - 所有用例都支持参数化

## 3. 数据与边界
- **正常数据集**: 2-5维张量，float32/float64类型，合理形状（如[3,4], [5,3,2]）
- **随机生成策略**: 固定随机种子，可重复测试
- **边界值**: 
  - 最小形状：[1,1]张量
  - 极端形状：超大/超小张量
  - 空输入：零元素张量（警告场景）
- **负例与异常场景**:
  - 维度<2的非法输入
  - 非张量类型输入
  - 参数越界：sparsity∉[0,1], std≤0
  - 维度不匹配：eye_非2D, dirac_非3-5D
  - 无效非线性函数参数

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖约束 | 优先级 |
|-------|----------|----------|--------|
| TC-01 | 基础分布初始化 | 张量维度≥2, 原地操作 | High |
| TC-02 | 正态分布初始化 | 统计分布验证 | High |
| TC-05 | Xavier初始化 | fan计算, 方差边界 | High |
| TC-06 | Kaiming初始化 | 非线性函数增益 | High |
| TC-09 | 特殊矩阵初始化 | 维度限制检查 | High |

**尚未覆盖的风险点**:
- 零维/一维张量处理未定义行为
- 缺少__all__定义的API边界模糊
- 随机性测试的统计稳定性
- 多设备（GPU）兼容性测试
- 所有20+个函数的完整参数组合

## 5. 迭代策略
1. **首轮（round1）**: 仅生成SMOKE_SET的5个用例，使用weak断言
2. **中间轮（roundN）**: 修复失败用例，逐步启用DEFERRED_SET
3. **最终轮（final）**: 启用strong断言，可选覆盖率检查

## 6. 文件结构
- 主文件: `tests/test_torch_nn_init.py`
- 分组文件: `tests/test_torch_nn_init_g1.py` (G1), `tests/test_torch_nn_init_g2.py` (G2), `tests/test_torch_nn_init_g3.py` (G3)
- 规格源: `test_plan.json`（机器可读）