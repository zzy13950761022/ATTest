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
        return """You are writing test requirements for the Python target `{target_fqn}`. Keep the document under 1200 words, retain only executable constraints, boundaries, exception scenarios, and coverage points, and do not copy long passages from `function_doc.md`.

Stage: Stage 2 - Define Requirements -> Output `requirements.md`

## Steps
1) Use `read_file` to read `function_doc.md` and extract parameters, return values, constraints, side effects, and risks.
2) Write `requirements.md` and cover the following content, preserving the structure and headings:

```markdown
# {target_fqn} Test Requirements

## 1. Goals and Scope
- Main functionality and expected behavior
- Out-of-scope content

## 2. Inputs and Constraints
- Parameter list (name, type/shape, default value)
- Valid value ranges, dimensionality, and device requirements
- Required and optional combinations
- Randomness and global-state requirements

## 3. Outputs and Evaluation
- Expected return structure and key fields
- Tolerance / error bounds (for example, floating point)
- State changes or side-effect checkpoints

## 4. Error and Exception Scenarios
- Exceptions or warnings triggered by invalid inputs, shapes, or types
- Boundary values (empty, None, zero length, extreme shapes/values)

## 5. Dependencies and Environment
- External resource, device, network, or file dependencies
- Parts that require mock/monkeypatch(you must provide concrete symbol paths, such as `torch.nn.parallel.scatter_gather.scatter_kwargs`)

## 6. Coverage and Priorities
- Mandatory paths (high priority, at most 5 short items)
- Optional paths (medium/low priority merged into one list)
- Known risks / missing information (list items only, no expansion)
```

Use `write_file` only to write `requirements.md`. Do not paste the full content into the conversation.
{user_feedback}
"""
    
    def get_config(self) -> StageConfig:
        return self.config
