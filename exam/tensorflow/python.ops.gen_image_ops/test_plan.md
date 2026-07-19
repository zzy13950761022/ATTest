# tensorflow.python.ops.gen_image_ops 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01, CASE_02, CASE_03, CASE_04, CASE_05
- DEFERRED_SET: CASE_06, CASE_07, CASE_08, CASE_09, CASE_10
- 测试文件路径：tests/test_tensorflow_python_ops_gen_image_ops.py（单文件）
- 断言分级策略：首轮使用 weak 断言，最终启用 strong 断言
- 预算策略：每个用例 size=S，max_lines=80，max_params=8

## 3. 数据与边界
- 正常数据集：随机生成符合形状约束的张量
- 边界值：空张量、零尺寸、极端形状（超大/超小）
- 数值边界：0, 255, -inf, inf, nan，有效数据类型范围
- 负例场景：维度不足、不支持类型、无效参数值
- 异常场景：None输入、越界索引、无效图像数据

## 4. 覆盖映射
- TC-01 (decode_jpeg)：覆盖图像编解码核心功能
- TC-02 (resize_bilinear)：覆盖尺寸调整和插值
- TC-03 (non_max_suppression)：覆盖边界框处理
- TC-04 (adjust_contrastv2)：覆盖颜色调整操作
- TC-05 (crop_and_resize)：覆盖裁剪和调整复合操作

## 5. 尚未覆盖的风险点
- 约50个函数中仅覆盖5个核心函数
- 依赖TensorFlow C++后端实现验证有限
- 机器生成代码可能被覆盖的风险
- 各函数边界条件需单独分析
- 缺少模块级使用示例参考