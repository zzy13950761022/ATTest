# tensorflow.python.util.nest 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures（用于C++扩展和复合张量）
- 随机性处理：固定随机种子/控制 RNG（不适用，无随机性）
- 测试类型：纯函数测试，无副作用

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: CASE_01 (flatten基本嵌套结构), CASE_02 (map_structure函数应用), CASE_03 (assert_same_structure验证)
- DEFERRED_SET: CASE_04 (pack_sequence_as结构重建), CASE_05 (is_nested嵌套判断)
- 测试文件路径：tests/test_tensorflow_python_util_nest.py（单文件）
- 断言分级策略：首轮使用weak断言，最终轮启用strong断言
- 预算策略：所有用例size=S，max_lines≤80，max_params≤6

## 3. 数据与边界
- 正常数据集：列表、元组、字典的嵌套组合，简单数值和字符串
- 随机生成策略：不适用，使用确定性测试数据
- 边界值：空结构[]、{}、()，单元素结构，混合类型结构
- 极端形状：接近递归限制的嵌套深度（后续扩展）
- 空输入：空列表、空字典、空元组
- 负例与异常场景：
  - 结构不匹配引发ValueError
  - 类型检查失败（check_types=True）
  - 序列长度不匹配
  - 非排序字典键处理

## 4. 覆盖映射
| TC ID | 对应需求 | 覆盖约束 | 优先级 |
|-------|----------|----------|--------|
| TC-01 | flatten基本功能 | 嵌套结构扁平化 | High |
| TC-02 | map_structure函数应用 | 结构保持的函数映射 | High |
| TC-03 | assert_same_structure验证 | 结构比较与类型检查 | High |
| TC-04 | pack_sequence_as结构重建 | 序列打包与结构恢复 | High |
| TC-05 | is_nested嵌套判断 | 嵌套结构识别 | High |

### 尚未覆盖的风险点
- 复合张量展开（expand_composites=True）
- 特殊Python类型处理（namedtuple、dataclass、attrs）
- 字典键排序行为验证
- 超大嵌套结构处理（递归深度限制）
- 循环引用行为（未定义，测试需避免）

## 5. 迭代策略
- 首轮（round1）：仅生成SMOKE_SET用例，使用weak断言
- 后续轮（roundN）：修复失败用例，从deferred_set提升用例
- 最终轮（final）：启用strong断言，可选覆盖率检查

## 6. Mock需求
当测试复合张量功能时，需要mock以下目标：
- tensorflow.python.util.nest._pywrap_utils
- tensorflow.python.util.nest._pywrap_nest  
- tensorflow.python.framework.composite_tensor
- tensorflow.python.framework.sparse_tensor
- tensorflow.python.ops.ragged.ragged_tensor