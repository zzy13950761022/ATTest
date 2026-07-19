"""
Stage 3: Design Test Plan
Creates a concrete test plan with specific test cases.
"""
from ..stage import Stage, StageConfig


class TestPlanStage(Stage):
    """
    Design specific test cases based on requirements.
    """
    
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="design_test_plan",
            display_name="Design Test Plan",
            description="Plan specific test cases and scenarios",
            prompt_template=self._get_prompt_template(),
            input_artifacts=["function_doc.md", "requirements.md"],
            output_artifacts=["test_plan.md", "test_plan.json"],
            tools=["read_file", "write_file"],
            allow_skip=False
        )
    
    def _get_prompt_template(self) -> str:
        return """Design a specification-driven test plan for `{target_fqn}` from a pytest perspective. `test_plan` is the sole source of truth for later test-case generation. In the first round, produce only a small number of runnable cases so later iterations can extend them.

Stage: Stage 3 - Design Test Plan -> Output `test_plan.md` and `test_plan.json`

## Steps
1) Use `read_file` to read `function_doc.md` and `requirements.md`.
2) Generate `test_plan.json` (machine-readable, the single source of specification truth) and `test_plan.md` (a short summary that references that specification).

## Key Constraints (Must Follow)
1) **Prioritize `SMOKE_SET`**: in the first round, generate only 3-5 core cases to ensure a minimal runnable set.
2) **Low/Medium must not create standalone CASE blocks**: they must be implemented as parameter extensions of existing High-priority CASE blocks.
3) **Assertion levels**: every CASE must declare weak and strong assertion lists; the first round uses weak assertions only.
4) **Budget control**: every CASE must declare `size`, `max_lines`, `max_params`, `is_parametrized`, and `requires_mock`.
5) **Fixed iteration strategy**: encode the strategies for the first round, later rounds, and the final round directly in the specification.
6) **Single-file by default**: output a single-file plan unless grouping is explicitly necessary. Do not generate `groups`, `active_group_order`, or `test_files.groups` by default.
7) **`BLOCK_ID` mapping**: each test case must map one-to-one to `CASE_XX`; `BLOCK_ID`s must remain stable and immutable.
8) **Explicit mock targets**: when `requires_mock=true`, you must fill `mock_targets` with full import paths and keep them consistent with the mock constraints in `requirements.md`.

## `test_plan.json` Structure (Strict JSON)
```json
{{
  "plan_version": 2,
  "target": "{target_fqn}",
  "block_rules": {{
    "header_block": "HEADER",
    "footer_block": "FOOTER",
    "case_prefix": "CASE_",
    "case_format": "CASE_01"
  }},
  "iteration_strategy": {{
    "round1": {{"include": "SMOKE_SET", "assert_level": "weak", "max_blocks": 5}},
    "roundN": {{"only_fix_failed_blocks": true, "block_limit": 3, "promote_deferred": true}},
    "final": {{"enable_strong_asserts": true, "coverage_optional": true}}
  }},
  "test_files": {{
    "default": "tests/test_{target_slug}.py",
    "all_pattern": "tests/test_{target_slug}.py"
  }},
  "cases": [
    {{
      "tc_id": "TC-01",
      "block_id": "CASE_01",
      "name": "core path",
      "priority": "High",
      "param_matrix": [
        {{"dtype": "float32", "device": "cpu", "shape": [2, 2], "flags": []}}
      ],
      "asserts": {{
        "weak": ["shape", "dtype", "finite", "basic_property"],
        "strong": ["approx_equal", "orthogonality"]
      }},
      "oracle": "torch.linalg.eigh",
      "assertion_level": "weak",
      "size": "S",
      "max_lines": 80,
      "max_params": 6,
      "is_parametrized": true,
      "requires_mock": false,
      "mock_targets": ["torch.nn.parallel.scatter_gather.scatter_kwargs"]
    }}
  ],
  "param_extensions": [
    {{
      "base_block_id": "CASE_01",
      "priority": "Medium",
      "params": {{"dtype": "float64", "device": "cpu", "shape": [4, 4], "flags": ["edge"]}},
      "note": "used as a parameter extension"
    }}
  ],
  "smoke_set": ["CASE_01", "CASE_02", "CASE_03"],
  "deferred_set": ["CASE_04", "CASE_05", "CASE_06"]
}}
```

## `test_plan.md` Structure (Short Summary)

```markdown
# {target_fqn} Test plan

## 1. Testing Strategy
- Unit test framework: pytest
- Isolation strategy: mock/monkeypatch/fixtures
- Randomness handling: fixed seed / controlled RNG

## 2. Generated Specification Summary (from test_plan.json)
- `SMOKE_SET`: list the included `CASE_XX` blocks
- `DEFERRED_SET`: list the deferred `CASE_XX` blocks
- Test file path (single-file output)
- Assertion-level strategy (`weak` / `strong`)
- Budget strategy (`size` / `max_lines` / `max_params`)

## 3. Data and Boundaries
- Normal datasets and random data generation strategy (short phrases)
- Boundary values, extreme shapes, and empty inputs (each item under 15 words)
- Negative and exception scenario list (titles only)

## 4. Coverage Mapping
- Requirement and constraint mapping for each test case (table or short list)
- Uncovered risks (key risks only)
```

**Use `write_file` only to write `test_plan.json` and `test_plan.md`**. Do not paste the full content into the conversation.
{user_feedback}
"""
    
    def get_config(self) -> StageConfig:
        return self.config
