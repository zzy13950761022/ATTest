# torch.nn.utils.rnn 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：使用fixtures管理测试数据，monkeypatch处理设备检测
- 随机性处理：固定随机种子，控制序列生成
- 设备兼容性：支持CPU测试，CUDA可选（通过环境变量控制）

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04（4个核心用例）
- **DEFERRED_SET**: CASE_05-CASE_10（6个延期用例）
- **group列表**:
  - G1: 核心打包解包函数族（pack_padded_sequence, pad_packed_sequence等）
  - G2: 填充序列处理函数族（pad_sequence, unpad_sequence）
  - G3: 边界与异常处理
- **active_group_order**: G1 → G2 → G3（按功能复杂度排序）
- **断言分级策略**: 首轮使用weak断言（类型、形状、设备等基础验证）
- **预算策略**: 每个用例size=S，max_lines≤80，max_params≤6

## 3. 数据与边界
- **正常数据集**: 随机生成变长序列，长度范围[1, 10]，batch_size范围[1, 5]
- **边界值**: 空列表、单元素序列、零长度、极大batch_size
- **极端形状**: 长序列（接近内存限制）、大batch_size
- **数据类型**: float32, float64, float16, int64
- **设备**: CPU（必需），CUDA（可选）
- **负例场景**: 
  - lengths与batch_size不匹配
  - enforce_sorted=True但未排序
  - total_length小于实际长度
  - 设备不匹配异常
  - 非张量输入异常

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖函数 | 关键验证点 |
|-------|----------|----------|------------|
| TC-01 | 必测路径1 | pack_padded_sequence | 基本打包功能，PackedSequence属性 |
| TC-02 | 必测路径1 | pad_packed_sequence | 逆操作恢复，数据一致性 |
| TC-03 | 必测路径4 | pad_sequence | 序列填充，padding值应用 |
| TC-04 | 必测路径2 | pack_padded_sequence | enforce_sorted参数行为 |
| TC-05 | 可选路径 | pack_sequence/unpack_sequence | 便捷函数功能 |

**尚未覆盖的风险点**:
- C++实现边界条件（依赖mock）
- 混合精度转换的数值稳定性
- 多线程环境行为
- 内存泄漏和性能问题
- 极端大尺寸序列处理