"""
Stage 2: Generate Requirements
Defines comprehensive test requirements based on function understanding.
"""
from ..stage import Stage, StageConfig


class RequirementsStage(Stage):
    """
    Generate detailed test requirements that define what needs to be tested.
    """
    
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="generate_requirements",
            display_name="Generate Requirements",
            description="Define comprehensive test requirements",
            prompt_template=self._get_prompt_template(),
            input_artifacts=["function_doc.md"],
            output_artifacts=["requirements.md"],
            tools=["read_file", "write_file"],
            allow_skip=False
        )
    
    def _get_prompt_template(self) -> str:
        return """你正在为 Python 目标 `{target_fqn}` 撰写测试需求说明。全文不超过 1200 字，只保留可执行约束/边界/异常/覆盖要点，禁止复制 function_doc 的长段落。

阶段：Stage 2 - 需求定义 → 输出 `requirements.md`

## 步骤
1) 用 `read_file` 读取 `function_doc.md`，提炼参数/返回/约束/副作用/风险。
2) 编写 `requirements.md`，覆盖以下内容（保持结构与标题）：

```markdown
# {target_fqn} 测试需求

## 1. 目标与范围
- 主要功能与期望行为
- 不在范围内的内容

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
- 有效取值范围/维度/设备要求
- 必需与可选组合
- 随机性/全局状态要求

## 3. 输出与判定
- 期望返回结构及关键字段
- 容差/误差界（如浮点）
- 状态变化或副作用检查点

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
- 边界值（空、None、0 长度、极端形状/数值）

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
- 需要 mock/monkeypatch 的部分（必须写出具体符号路径，如 `torch.nn.parallel.scatter_gather.scatter_kwargs`）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
- 可选路径（中/低优先级合并为一组列表）
- 已知风险/缺失信息（仅列条目，不展开）
```

只用 `write_file` 写入 `requirements.md`，不要在对话中粘贴全文。
{user_feedback}
"""
    
    def get_config(self) -> StageConfig:
        return self.config
