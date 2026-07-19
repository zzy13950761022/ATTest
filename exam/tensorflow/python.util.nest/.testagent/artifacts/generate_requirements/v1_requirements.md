# tensorflow.python.util.nest 测试需求

## 1. 目标与范围
- 主要功能与期望行为：测试嵌套结构处理模块，验证结构扁平化、映射、比较、打包等核心操作的正确性
- 不在范围内的内容：TensorFlow 特定张量操作、第三方库集成、性能基准测试

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - `structure`: 嵌套结构或原子值（Python 集合或原子类型）
  - `shallow_tree`: 浅层结构（用于部分操作）
  - `func`: 可调用对象（用于 map_structure）
  - `check_types`: bool（默认 True，是否检查类型一致性）
  - `expand_composites`: bool（默认 False，是否展开复合张量）
- 有效取值范围/维度/设备要求：
  - 嵌套深度：任意但有限（避免递归深度限制）
  - 结构必须形成树，禁止循环引用
  - 字典键必须可排序以确保确定性
- 必需与可选组合：
  - flatten/structure 必需
  - map_structure 需要 func 参数
  - assert_same_structure 需要两个结构参数
- 随机性/全局状态要求：无随机性，无全局状态修改

## 3. 输出与判定
- 期望返回结构及关键字段：
  - flatten: 返回扁平化列表，保持原始顺序
  - map_structure: 返回应用函数后的新结构，保持原结构
  - assert_same_structure: 无返回值，失败时抛出 ValueError
  - pack_sequence_as: 返回按指定结构打包的序列
- 容差/误差界（如浮点）：不适用（无数值计算）
- 状态变化或副作用检查点：无副作用，纯函数操作

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - 循环引用结构：未定义行为，测试需避免
  - 非排序字典键：可能引发异常
  - 类型不匹配（check_types=True）：ValueError
  - 结构不匹配：ValueError
- 边界值（空、None、0 长度、极端形状/数值）：
  - 空结构：[]、{}、()
  - None 值处理
  - 单元素结构
  - 超大嵌套深度（接近递归限制）
  - 混合类型结构

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：无
- 需要 mock/monkeypatch 的部分：
  - `tensorflow.python.util.nest._pywrap_utils`（C++ 扩展）
  - `tensorflow.python.util.nest._pywrap_nest`（C++ 扩展）
  - `tensorflow.python.framework.composite_tensor`（复合张量支持）
  - `tensorflow.python.framework.sparse_tensor`（稀疏张量）
  - `tensorflow.python.ops.ragged.ragged_tensor`（不规则张量）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）：
  1. flatten 基本嵌套结构（列表、元组、字典混合）
  2. map_structure 函数应用保持结构
  3. assert_same_structure 类型检查与结构验证
  4. pack_sequence_as 序列打包与结构重建
  5. is_nested 嵌套判断正确性
- 可选路径（中/低优先级合并为一组列表）：
  - 复合张量展开（expand_composites=True）
  - 特殊 Python 类型处理（namedtuple、dataclass、attrs）
  - 字典键排序行为验证
  - 超大嵌套结构处理
  - 错误消息格式与内容
  - 性能边界情况（递归深度限制）
- 已知风险/缺失信息（仅列条目，不展开）：
  - 循环引用行为未定义
  - 类型注解信息不完整
  - 特殊 Python 对象处理边界
  - 复合张量展开的完整覆盖
  - 字典非排序键的完全处理