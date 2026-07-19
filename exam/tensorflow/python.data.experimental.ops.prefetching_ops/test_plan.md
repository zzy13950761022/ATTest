# tensorflow.python.data.experimental.ops.prefetching_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（GPU设备可用性、内存分配）
- 随机性处理：固定随机种子，使用确定性数据集生成
- 设备隔离：CPU测试为主，GPU测试需要mock环境

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_05（3个核心用例）
- **DEFERRED_SET**: CASE_03, CASE_04, CASE_06, CASE_07（4个扩展用例）
- **group列表**: 
  - G1: prefetch_to_device核心功能（CASE_01-04）
  - G2: copy_to_device与设备间传输（CASE_05-07）
- **active_group_order**: G1, G2
- **断言分级策略**: 首轮使用weak断言，最终轮启用strong断言
- **预算策略**: 
  - size: S（小型测试）
  - max_lines: 60-75行
  - max_params: 4-5个参数
  - 所有用例均为参数化测试

## 3. 数据与边界
- **正常数据集**: 小规模tensor_slices（5-10个元素），int32/float32/float64类型
- **边界值**: 
  - buffer_size=None（自动选择）
  - 空数据集（0元素）
  - 相同源和目标设备
  - 多维数据形状（2x3x4）
- **负例与异常场景**:
  - 无效设备字符串
  - 负值buffer_size
  - 不存在的GPU设备
  - 非字符串设备参数

## 4. 覆盖映射
| TC_ID | 需求覆盖 | 约束覆盖 |
|-------|----------|----------|
| TC-01 | prefetch_to_device基本功能验证 | device参数必需，返回可调用函数 |
| TC-02 | buffer_size参数验证 | buffer_size可选，自动选择机制 |
| TC-03 | 无效参数处理 | 无效设备字符串异常 |
| TC-04 | 空数据集处理 | 边界值处理 |
| TC-05 | copy_to_device基本功能验证 | 设备间数据传输 |
| TC-06 | GPU支持测试 | GPU设备支持，initializable_iterator要求 |
| TC-07 | 无效设备处理 | 错误处理机制 |

## 5. 尚未覆盖的风险点
- buffer_size自动选择机制未详细说明
- 多GPU环境下的行为未详细描述
- 内存不足场景处理机制
- 并发访问测试
- 大数据集性能影响

## 6. 迭代策略
- **首轮**: 仅生成SMOKE_SET（3个用例），使用weak断言
- **后续轮**: 修复失败用例，逐步添加DEFERRED_SET
- **最终轮**: 启用strong断言，可选覆盖率检查

## 7. 文件结构
- 主测试文件: `tests/test_tensorflow_python_data_experimental_ops_prefetching_ops.py`
- 分组文件: 
  - G1: `tests/test_tensorflow_python_data_experimental_ops_prefetching_ops_g1.py`
  - G2: `tests/test_tensorflow_python_data_experimental_ops_prefetching_ops_g2.py`