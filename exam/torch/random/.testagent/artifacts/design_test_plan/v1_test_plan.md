# torch.random 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（针对CUDA依赖）
- 随机性处理：固定随机种子/控制RNG状态序列
- 模块拆分：3个功能组分别测试

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04（4个核心用例）
- **DEFERRED_SET**: CASE_05-CASE_12（8个延期用例）
- **group列表**:
  - G1: 种子管理与状态操作（manual_seed, seed, initial_seed）
  - G2: 状态保存与恢复（set_rng_state, get_rng_state）
  - G3: 上下文隔离与设备管理（fork_rng）
- **active_group_order**: G1 → G2 → G3
- **断言分级策略**: 首轮仅使用weak断言，最终轮启用strong断言
- **预算策略**: 
  - size: S（小型用例）
  - max_lines: 60-75行
  - max_params: 3-6个参数

## 3. 数据与边界
- **正常数据集**: 标准种子值(42)、有效状态张量、默认上下文
- **随机生成策略**: 使用固定种子确保可重复性
- **边界值**:
  - manual_seed: -0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff
  - 空状态张量处理
  - 负种子自动映射
- **极端形状**: 最小/最大有效状态尺寸
- **空输入**: fork_rng空设备列表
- **负例与异常场景**:
  - 非ByteTensor状态输入
  - 非法设备ID
  - 错误形状状态张量
  - 非整数种子类型

## 4. 覆盖映射
- **TC-01**: manual_seed基本功能 → 需求5.1
- **TC-02**: seed/initial_seed基本功能 → 需求5.4, 5.5
- **TC-03**: 状态保存恢复 → 需求5.2
- **TC-04**: fork_rng上下文 → 需求5.3
- **TC-05**: manual_seed边界值 → 需求4.2
- **TC-06**: 状态操作异常 → 需求4.1

- **尚未覆盖的风险点**:
  - 多CUDA设备并发操作
  - 混合CPU/CUDA状态同步
  - 大状态张量性能
  - 负种子映射公式验证
  - 多设备警告触发条件