# tensorflow.python.ops.sets_impl 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures 用于底层 gen_set_ops 操作
- 随机性处理：固定随机种子控制稀疏张量生成
- 测试层级：模块级单元测试，覆盖所有公共 API 函数

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01（稀疏张量基本集合操作）、CASE_02（密集-稀疏张量混合操作）、CASE_03（集合大小计算）
- **DEFERRED_SET**: CASE_04（集合差集方向控制）、CASE_05（索引验证开关测试）
- **测试文件路径**: tests/test_tensorflow_python_ops_sets_impl.py（单文件）
- **断言分级策略**: 首轮使用 weak 断言（形状、类型、基本结构），后续启用 strong 断言（精确值、完整覆盖）
- **预算策略**: 每个用例 size=S，max_lines=70-85，max_params=5-7

## 3. 数据与边界
- **正常数据集**: 随机生成稀疏张量，密度 0.3-0.7，支持 int32/int64 类型
- **边界值**: 
  - 空集合（零元素稀疏张量）
  - 单元素集合边界
  - 高维稀疏张量（3D+形状）
  - int64 边界值测试
  - 字符串类型元素（后续扩展）
- **负例与异常场景**:
  - 不支持的数据类型
  - 稀疏索引未排序（validate_indices=True）
  - 输入维度不匹配
  - 不支持 SparseTensor,DenseTensor 顺序
  - 无效稀疏张量结构

## 4. 覆盖映射
| TC_ID | 对应需求/约束 | 覆盖函数 |
|-------|--------------|----------|
| TC-01 | 稀疏-稀疏基本操作 | set_intersection |
| TC-02 | 密集-稀疏混合操作 | set_union |
| TC-03 | 集合大小计算 | set_size |
| TC-04 | 差集方向控制 | set_difference |
| TC-05 | 索引验证开关 | 所有函数（validate_indices参数） |

**尚未覆盖的风险点**:
- 字符串类型集合操作的具体行为
- 底层 C++ 实现细节未暴露
- 不支持 SparseTensor,DenseTensor 顺序的具体原因
- 内存使用和性能约束未定义
- 高并发场景下的线程安全性