# tensorflow.python.ops.custom_gradient 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 测试装饰器函数正确包装用户函数，保持原始输出值
  - 验证自定义梯度函数正确计算和传播梯度
  - 确保图模式和急切执行模式行为一致
  - 测试变量梯度计算（ResourceVariable类型）
  - 验证嵌套自定义梯度场景
- 不在范围内的内容
  - 其他模块函数（recompute_grad、grad_pass_through）
  - 非TensorFlow操作的自定义梯度
  - 第三方库集成测试

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - f: 函数/None，默认值None，支持装饰器语法
- 有效取值范围/维度/设备要求
  - 被装饰函数必须返回元组 (y, grad_fn)
  - y: TensorFlow张量或嵌套结构
  - grad_fn: 函数签名 g(*grad_ys) 或 g(*grad_ys, variables=None)
  - 变量必须是ResourceVariable类型
- 必需与可选组合
  - 必需：被装饰函数返回正确格式的元组
  - 可选：f参数可为None（返回装饰器工厂）
- 随机性/全局状态要求
  - 无随机性要求
  - 全局状态：梯度计算图修改

## 3. 输出与判定
- 期望返回结构及关键字段
  - 装饰后函数h(x)返回与f(x)[0]相同的值
  - 梯度计算由f(x)[1]定义
  - 返回类型：装饰器包装的函数
- 容差/误差界（如浮点）
  - 浮点误差：1e-6相对误差
  - 梯度数值稳定性验证
- 状态变化或副作用检查点
  - 梯度计算图正确修改
  - 变量梯度正确传播
  - 无意外副作用

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 被装饰函数不返回元组
  - grad_fn签名不正确
  - 非ResourceVariable类型变量
  - 梯度数量不匹配
- 边界值（空、None、0长度、极端形状/数值）
  - f=None边界：装饰器工厂模式
  - 空张量输入
  - 极端数值梯度（如log1pexp数值稳定性）
  - 零梯度场景

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - TensorFlow库依赖
  - GPU/CPU设备可用性
- 需要mock/monkeypatch的部分
  - `tensorflow.python.eager.tape_lib.record_operation`
  - `tensorflow.python.framework.ops.RegisterGradient`
  - `tensorflow.python.ops.gradients_util._get_dependent_variables`
  - `tensorflow.python.ops.gradients_util.get_variable_by_name`

## 6. 覆盖与优先级
- 必测路径（高优先级，最多5条，短句）
  1. 基本装饰器功能：函数包装和梯度计算
  2. 图模式与急切模式一致性验证
  3. ResourceVariable变量梯度传播
  4. 嵌套自定义梯度场景
  5. 数值稳定性边界测试（log1pexp示例）
- 可选路径（中/低优先级合并为一组列表）
  - 多变量输入场景
  - 停止梯度行为
  - 关键字参数支持（急切模式）
  - 复杂嵌套结构输出
  - 二阶梯度计算
  - 装饰器工厂模式（f=None）
- 已知风险/缺失信息（仅列条目，不展开）
  - 非ResourceVariable变量处理
  - grad_fn返回梯度数量验证细节
  - 嵌套自定义梯度二阶行为
  - 图模式kwargs限制