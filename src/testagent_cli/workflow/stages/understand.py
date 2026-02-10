"""
Stage 1: Understand Function
Analyzes the operator semantics and API signature.
"""
from ..stage import Stage, StageConfig


class UnderstandFunctionStage(Stage):
    """
    Analyze the target operator to understand its functionality,
    parameters, and constraints.
    """
    
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="understand_function",
            display_name="Understand Function",
            description="Analyze Python target semantics and API signature",
            prompt_template=self._get_prompt_template(),
            input_artifacts=[],  # First stage, no inputs
            output_artifacts=["function_doc.md"],
            tools=["inspect_python", "list_files", "read_file", "search", "write_file"],  # Tools for exploration
            allow_skip=False
        )
    
    def _get_prompt_template(self) -> str:
        return """你是一个 Python 代码分析助手，目标是为指定函数生成测试。请控制全文不超过 800 字，使用短句/列表呈现要点，避免重复。

当前阶段：**理解目标函数**（Stage 1） → 产物 `function_doc.md`

- 目标 FQN: {target_fqn}
- 工作目录/项目根: {project_root}
- 预期测试文件路径: {test_file_path}

## 需要做什么
1) 使用 `inspect_python` 获取目标的签名、注解、docstring、源码片段、模块路径。
   - 必须传递以下参数（JSON 格式）：
     ```json
     {{
       "target": "{target_fqn}",
       "add_cwd_to_path": true,
       "max_source_length": 4000
     }}
     ```
   - 目标可能是 `pkg.module:func`、`pkg.module.Class.method`，也可能是一个模块
   - 如果目标是模块：先列出模块导出的核心类/函数（可根据 `__all__` 或主要公共 API），再聚焦最核心的主类/函数做说明，并在"风险与空白"注明多实体情况
2) 如有必要，用 `read_file` 查看源码文件/README 以补充约束（如张量形状、dtype、示例）。
3) 生成 `function_doc.md`，格式如下（必须包含所有字段）：

```markdown
# {target_fqn} - 函数说明

## 1. 基本信息
- **FQN**: {target_fqn}
- **模块文件**: `path/to/module.py`
- **签名**: func(param1, param2=..., *args, **kwargs)
- **对象类型**: function/method/class/callable

## 2. 功能概述
用 2-3 句话描述行为与返回值。

## 3. 参数说明
- name (类型/默认值): 约束、形状/范围、是否可选
- ...

## 4. 返回值
- 类型/结构、关键字段、可能的 None/异常

## 5. 文档要点
- docstring 中的重要约束（张量形状、dtype、设备等）
- 预期前置条件/后置条件

## 6. 源码摘要
- 关键路径/分支（最多 5 条），依赖的辅助函数或外部 API
- 副作用（I/O、随机性、全局状态），只列事实，不展开叙述

## 7. 示例与用法（如有）
- 来源：docstring/源码/示例

## 8. 风险与空白
- 未提供的类型信息、模糊/缺失的约束
- 需要在测试中特别覆盖的边界
- 缺少信息时直接列在此处，不要在正文重复描述
```

只通过 `write_file` 输出 `function_doc.md`，不要直接在对话中粘贴完整文档。
如果找不到 docstring/源码，要在文档的“风险与空白”中明确说明。

## 用户反馈
{user_feedback}
"""
    
    def get_config(self) -> StageConfig:
        return self.config
