# tensorflow.python.data.experimental.ops.readers 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock文件I/O、数据库连接、压缩操作
- 随机性处理：固定随机种子，控制文件读取顺序

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04
- DEFERRED_SET: CASE_05, CASE_06, CASE_07, CASE_08
- group列表：G1(CSV), G2(SQL), G3(TFRecord)
- active_group_order: G1 → G2 → G3
- 断言分级策略：首轮使用weak断言（类型、形状、基本属性）
- 预算策略：每个CASE最多80行，6个参数，S大小

## 3. 数据与边界
- 正常数据：模拟CSV、SQL、TFRecord文件内容
- 边界值：空文件、单行文件、极大batch_size、特殊字符
- 极端形状：超大数值、嵌套结构、多文件列表
- 空输入：None参数、空列表、零长度字符串
- 负例场景：无效文件路径、不支持的数据类型、错误压缩格式

## 4. 覆盖映射
| TC ID | 对应功能 | 覆盖需求 | 风险点 |
|-------|----------|----------|--------|
| TC-01 | make_csv_dataset | CSV基本读取、批处理、类型推断 | 类型推断逻辑覆盖不足 |
| TC-02 | CsvDataset | CSV记录解析、列选择、类型转换 | 特殊字符处理边界 |
| TC-03 | SqlDataset | SQL查询执行、结果解析、连接管理 | 线程安全性和资源泄漏 |
| TC-04 | make_tf_record_dataset | TFRecord读取、批处理、压缩支持 | 序列化完整性验证 |

**尚未覆盖的关键风险点**：
- 并行读取线程安全性
- 内存使用特性基准
- 大型数据集分片支持
- 压缩文件错误恢复机制
- V1/V2版本兼容性细节