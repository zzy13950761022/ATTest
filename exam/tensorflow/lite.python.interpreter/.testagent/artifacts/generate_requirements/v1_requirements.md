# tensorflow.lite.python.interpreter 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证Interpreter类正确加载TFLite模型、执行推理、管理张量、支持签名运行器
- 不在范围内的内容：模型训练、转换工具、移动端部署、自定义操作实现

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）：
  - model_path: str/None，TFLite文件路径
  - model_content: bytes/None，模型二进制内容
  - experimental_delegates: list/None，委托对象列表
  - num_threads: int/None，线程数（>=1）
  - experimental_op_resolver_type: OpResolverType枚举
  - experimental_preserve_all_tensors: bool，默认False
- 有效取值范围/维度/设备要求：
  - model_path与model_content至少一个非None
  - num_threads >= -1（-1表示自动）
  - 委托仅支持CPython实现
- 必需与可选组合：
  - model_path或model_content必须提供其一
  - 其他参数均为可选
- 随机性/全局状态要求：
  - 无随机性要求
  - 加载共享库可能修改全局状态

## 3. 输出与判定
- 期望返回结构及关键字段：
  - get_input_details()/get_output_details(): 返回字典列表，包含index、name、shape、dtype、quantization
  - get_tensor(): 返回numpy数组
  - invoke(): 无返回值，执行推理
- 容差/误差界（如浮点）：
  - 浮点推理结果允许1e-5相对误差
  - 量化模型需验证scale/zero_point正确性
- 状态变化或副作用检查点：
  - allocate_tensors()后内存分配完成
  - set_tensor()后输入数据就绪
  - invoke()后输出张量可访问

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告：
  - model_path和model_content同时为None
  - 无效模型文件路径或损坏内容
  - 张量索引越界
  - 输入数据shape/dtype不匹配
  - 未调用allocate_tensors()直接invoke()
- 边界值（空、None、0长度、极端形状/数值）：
  - 空模型文件
  - 零长度输入张量
  - 超大shape导致内存溢出
  - num_threads=0或负值（除-1外）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - 需要有效的TFLite模型文件
  - 依赖C++共享库（_interpreter_wrapper）
  - 需要numpy进行数组操作
- 需要mock/monkeypatch的部分：
  - 文件系统访问（model_path）
  - 委托加载失败场景
  - 内存分配失败异常

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）：
  1. 使用model_path和model_content分别加载模型
  2. 完整推理流程：allocate_tensors→set_tensor→invoke→get_tensor
  3. 输入输出张量详细信息获取与验证
  4. 多线程配置（num_threads参数）
  5. 签名运行器基本功能
- 可选路径（中/低优先级合并为一组列表）：
  - 委托功能测试（仅CPython环境）
  - 中间张量保留功能
  - 操作解析器类型切换
  - 异常恢复与重试
  - 内存泄漏检测
  - 并发访问测试
- 已知风险/缺失信息（仅列条目，不展开）：
  - 缺少具体测试模型文件
  - 委托功能平台限制
  - 内存管理依赖Python GC
  - 中间张量访问可能未定义
  - 缺少错误处理具体示例