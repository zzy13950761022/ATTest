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
        return """You are a Python code analysis assistant. Your goal is to prepare testing context for the specified function. Keep the document under 800 words, use short sentences and bullet points, and avoid repetition.

Current stage: **Understand the target function** (Stage 1) -> Artifact: `function_doc.md`

- Target FQN: {target_fqn}
- Workspace / project root: {project_root}
- Expected test file path: {test_file_path}

## What To Do
1) Use `inspect_python` to obtain the target signature, annotations, docstring, source excerpt, and module path.
   - You must pass the following parameters (JSON):
     ```json
     {{
       "target": "{target_fqn}",
       "add_cwd_to_path": true,
       "max_source_length": 4000
     }}
     ```
   - The target may be `pkg.module:func`, `pkg.module.Class.method`, or an entire module.
   - If the target is a module, first list its core exported classes/functions (for example from `__all__` or the main public API), then focus on the primary class/function and note the multi-entity situation in the Risks and Gaps section.
2) If necessary, use `read_file` to inspect the source file or README for additional constraints such as tensor shapes, dtypes, or examples.
3) Generate `function_doc.md` in the following format, including every field:

```markdown
# {target_fqn} - Function documentation

## 1. Basic Information
- **FQN**: {target_fqn}
- **Module file**: `path/to/module.py`
- **Signature**: `func(param1, param2=..., *args, **kwargs)`
- **Object type**: function / method / class / callable

## 2. Functional Overview
Describe the behavior and return values in 2-3 sentences.

## 3. Parameter Description
- `name` (type / default): constraints, shape/range, whether it is optional
- ...

## 4. Return Values
- Type/structure, key fields, and possible `None` / exception behavior

## 5. Documentation Highlights
- Important constraints from the docstring (tensor shapes, dtypes, devices, and so on)
- Expected preconditions and postconditions

## 6. Source Summary
- Key paths/branches (at most 5), plus dependent helper functions or external APIs
- Side effects (I/O, randomness, global state); list facts only without elaboration

## 7. Examples and Usage (If Any)
- Source: docstring / source / examples

## 8. Risks and Gaps
- Missing type information and unclear or absent constraints
- Boundaries that require explicit testing coverage
- If information is missing, list it here directly and do not repeat it in the main body
```

Output `function_doc.md` only via `write_file`. Do not paste the full document directly into the conversation.
If the docstring or source cannot be found, state that explicitly in the Risks and Gaps section.

## User feedback
{user_feedback}
"""
    
    def get_config(self) -> StageConfig:
        return self.config
