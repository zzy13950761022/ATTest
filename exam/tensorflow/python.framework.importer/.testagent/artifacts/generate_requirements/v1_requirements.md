# tensorflow.python.framework.importer 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证 GraphDef 协议缓冲区正确导入到当前默认图，支持输入重映射和返回指定元素
- 不在范围内的内容：不测试已弃用的 op_dict 参数，不验证 producer_op_list 的具体业务逻辑

## 2. 输入与约束
- 参数列表：
  - graph_def: GraphDef proto 类型，必需
  - input_map: dict/None，字符串到 Tensor 对象的映射，可选
  - return_elements: list/None，字符串列表，可选
  - name: str/None，名称前缀，可选，默认 "import"
  - op_dict: dict/None，已弃用，不应使用
  - producer_op_list: OpList proto/None，可选
- 有效取值范围/维度/设备要求：
  - graph_def 必须是有效的 GraphDef 协议缓冲区
  - input_map 键必须是 graph_def 中存在的名称
  - return_elements 名称必须在 graph_def 中存在
- 必需与可选组合：graph_def 必需，其他参数可选
- 随机性/全局状态要求：操作修改当前默认图，需考虑图状态隔离

## 3. 输出与判定
- 期望返回结构及关键字段：
  - return_elements 不为 None 时：返回对应的 Operation 和/或 Tensor 对象列表
  - return_elements 为 None 时：返回 None
- 容差/误差界：不适用（非数值计算）
- 状态变化或副作用检查点：
  - 默认图中添加新操作和张量
  - 可能修改传入的 graph_def（添加属性默认值）
  - 返回元素顺序与请求顺序一致

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常：
  - 无效 GraphDef proto 类型
  - input_map 键不存在于 graph_def
  - return_elements 名称不存在
  - 非字符串类型的 return_elements 元素
- 边界值：
  - 空 GraphDef
  - input_map 为空字典
  - return_elements 为空列表
  - name 为 None 或空字符串
  - 控制输入映射（以 '^' 开头的名称）
  - 张量名称格式："operation_name" 和 "operation_name:output_index"

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：无
- 需要 mock/monkeypatch 的部分：
  - ops.get_default_graph() 用于图状态隔离
  - c_api.TF_GraphImportGraphDefWithResults 用于验证 C API 调用
  - 图操作创建和验证

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. 基本 GraphDef 导入，无 input_map 和 return_elements
  2. 带 input_map 的导入，验证输入重映射
  3. 带 return_elements 的导入，验证返回对象列表
  4. 无效 GraphDef 触发异常
  5. 名称冲突和重复导入场景
- 可选路径（中/低优先级）：
  - producer_op_list 参数的各种场景
  - 复杂嵌套图结构导入
  - 大规模图导入性能
  - 控制输入映射（^前缀名称）
  - 张量名称格式解析边界
  - 空图和最小图导入
- 已知风险/缺失信息：
  - producer_op_list 具体使用场景不明确
  - 缺少具体错误处理边界示例
  - 模块包含辅助函数 import_graph_def_for_function 未覆盖