# tensorflow.python.compiler.xla.jit 测试需求

## 1. 目标与范围
- 验证 `experimental_jit_scope` 上下文管理器正确控制 XLA JIT 编译行为
- 确保作用域内操作符按预期启用/禁用编译
- 不在范围内：具体编译性能优化效果、XLA 编译器内部实现细节

## 2. 输入与约束
- compile_ops: bool 或 callable，默认 True
  - bool: True/False 控制编译启用/禁用
  - callable: 接受 `node_def` 参数返回 bool 的函数
- separate_compiled_gradients: bool，默认 False
  - True: 每个梯度子图放入单独编译作用域
- 设备要求：必须在 graph execution 模式下，eager execution 模式不支持
- 全局状态：使用 `_XLA_SCOPE_KEY` 集合跟踪作用域深度

## 3. 输出与判定
- 返回上下文管理器对象，无显式返回值
- 进入作用域时设置编译属性，退出时恢复原设置
- 状态变化：作用域内操作符编译行为改变
- 副作用：全局作用域计数器更新

## 4. 错误与异常场景
- eager execution 模式下调用抛出 RuntimeError
- compile_ops 参数类型错误（非 bool/非 callable）
- 嵌套作用域深度异常
- 边界值：compile_ops=None, separate_compiled_gradients=None
- 极端形状/数值：不适用（无张量参数）

## 5. 依赖与环境
- TensorFlow 运行时环境
- XLA 编译器可用性
- 需要 mock: eager execution 检测逻辑
- 需要监控: 全局状态 `_XLA_SCOPE_KEY` 变化
- 依赖: graph execution 模式

## 6. 覆盖与优先级
- 必测路径：
  1. 基本上下文管理器功能验证
  2. eager execution 模式异常抛出
  3. compile_ops bool 参数功能测试
  4. compile_ops callable 参数条件编译
  5. separate_compiled_gradients 梯度分离

- 可选路径：
  - 嵌套作用域组合测试
  - 作用域退出后状态恢复验证
  - 不同 TensorFlow 版本兼容性
  - 与 tf.function 装饰器组合使用
  - 多线程环境下的作用域管理

- 已知风险/缺失信息：
  - 编译行为是"尽力而为"的，无保证
  - 缺少类型注解
  - 具体编译性能影响未量化
  - 作用域外操作可能被聚类编译
  - 实验性功能，API 可能变更