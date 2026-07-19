# torch.jit._trace - 函数说明

## 1. 基本信息
- **FQN**: torch.jit._trace:trace
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/jit/_trace.py`
- **签名**: trace(func, example_inputs, optimize=None, check_trace=True, check_inputs=None, check_tolerance=1e-05, strict=True, _force_outplace=False, _module_class=None, _compilation_unit=<torch.jit.CompilationUnit object>)
- **对象类型**: function

## 2. 功能概述
将函数或模块转换为可执行的 TorchScript ScriptFunction 或 ScriptModule。通过运行示例输入记录张量操作，生成优化后的 JIT 编译代码。适用于仅操作张量及其容器的代码。

## 3. 参数说明
- func (callable/torch.nn.Module): 要追踪的 Python 函数或 PyTorch 模块。参数和返回值必须是张量或包含张量的嵌套元组。
- example_inputs (tuple/torch.Tensor): 追踪时使用的示例输入元组。单个张量会自动包装为元组。
- check_trace (bool, 默认 True): 是否验证追踪代码与原始函数输出一致。
- check_inputs (list of tuples, 可选): 用于验证的额外输入参数列表。
- check_tolerance (float, 默认 1e-5): 验证时的浮点数比较容差。
- strict (bool, 默认 True): 是否在严格模式下运行追踪器。
- _force_outplace (bool, 默认 False): 内部参数，强制使用 outplace 操作。
- _module_class (可选): 内部参数，自定义模块类。
- _compilation_unit (CompilationUnit): 内部参数，编译单元对象。

## 4. 返回值
- 如果 func 是 nn.Module 或其 forward 方法：返回包含追踪代码的 ScriptModule 对象。
- 如果 func 是独立函数：返回 ScriptFunction 对象。
- 返回对象具有与原始模块相同的子模块和参数。

## 5. 文档要点
- 仅适用于无数据依赖的代码（无基于张量数据的条件判断）。
- 不能追踪控制流（if 语句、循环）。
- 不支持未跟踪的外部依赖（I/O、全局变量访问）。
- 训练/评估模式行为固定为追踪时的模式。
- 对于可变容器类型（list/dict），仅在非严格模式下记录。

## 6. 源码摘要
- 关键路径：调用内部追踪机制记录张量操作。
- 依赖辅助函数：_flatten, _unflatten 处理输入输出。
- 依赖外部 API：torch._C._jit_flatten, torch._C._jit_unflatten。
- 副作用：可能产生警告（TracerWarning），不修改全局状态。
- 使用 ONNXTracedModule 包装内部函数进行追踪。

## 7. 示例与用法
```python
# 追踪函数
def foo(x, y):
    return 2 * x + y
traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

# 追踪模块
class Net(nn.Module):
    def forward(self, x):
        return self.conv(x)
n = Net()
module = torch.jit.trace(n, torch.rand(1, 1, 3, 3))
```

## 8. 风险与空白
- 目标 FQN `torch.jit._trace` 是模块而非函数，包含多个实体（trace, trace_module, ONNXTracedModule 等）。
- 类型注解不完整：func 参数缺少具体类型约束。
- 未明确说明支持的张量 dtype 和设备限制。
- 缺少对复杂嵌套结构（如自定义类）的追踪支持说明。
- 内部参数（_force_outplace, _module_class, _compilation_unit）文档缺失。
- 需要特别测试边界：非确定性操作、动态控制流、训练/评估模式切换。
- 缺少对异常情况的详细说明（如无效输入、不支持的 Python 特性）。