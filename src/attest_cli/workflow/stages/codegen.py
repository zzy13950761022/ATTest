"""
Stage 4: Generate Code
Creates test code files and build/run scripts.
"""
import json
from pathlib import Path

from ...block_utils import build_block_index_json
from ..stage import Stage, StageConfig


class CodeGenStage(Stage):
    """
    Generate complete test code based on requirements and test plan.
    """
    
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        
        self.config = StageConfig(
            name="generate_code",
            display_name="Generate Code",
            description="Generate pytest cases for Python target",
            prompt_template=self._get_prompt_template(),
            input_artifacts=["function_doc.md", "requirements.md", "test_plan.md"],
            output_artifacts=[],  # LLM will write to project path directly
            tools=[
                "inspect_python",
                "list_files",
                "read_file",
                "part_read",
                "search",
                "write_file",
                "replace_in_file",
                "replace_block",
            ],  # Tools for code exploration and generation
            allow_skip=False
        )
    
    def _get_prompt_template(self) -> str:
        return """You are generating pytest test code for the Python target `{target_fqn}`.

Current stage: Stage 4 - Generate Test Code -> Target file `{test_file_path}`

## Iteration State
- epoch: {epoch_current}/{epoch_total}
- analysis_plan: {analysis_plan_status}
- active_group: {active_group_hint}

## Core Constraints (Must Follow)
1) Semantic blocks: BLOCK types are fixed as `HEADER` / `CASE_*` / `FOOTER`. Each `CASE` represents one test case or one parametrized group.
   - Marker format:
     - `# ==== BLOCK:HEADER START ====`
     - `# ==== BLOCK:HEADER END ====`
     - `# ==== BLOCK:CASE_01 START ====`
     - `# ==== BLOCK:CASE_01 END ====`
     - `# ==== BLOCK:FOOTER START ====`
     - `# ==== BLOCK:FOOTER END ====`
   - `BLOCK_ID`s must remain stable. Do not rename or reorder existing BLOCKs.
2) Edit budget: for any `BLOCK_ID`, allow at most one read and one replacement.
   - Prefer `replace_block` to avoid repeated `read_file` / `part_read` / `search` calls.
   - If the block still fails after one replacement, hand the decision back to `analyze_results`.
3) Single-write limit: each `write_file` / `replace_block` payload must be <= 8KB.
   - If content exceeds the limit, split it into multiple CASE blocks such as `CASE_03A` / `CASE_03B`, or reduce it through parametrization.
4) Incremental iteration: modify only the problematic BLOCKs listed in the analysis plan (1-3 per round).
5) No self-looping: stop after the current replacement round; do not perform repeated generation loops inside `generate_code`.

## Inputs
- Required: `requirements.md` and `test_plan.md`
- **Prefer the specification file**: if `test_plan.json` exists, treat it as the sole generation source; otherwise infer an equivalent structure from `test_plan.md`.
- Recommended: first read `function_doc.md`, `requirements.md`, and `test_plan.md` with `read_file` to understand the constraints.
- For iterative fixes, prefer reading the analysis plan at these paths relative to `{project_root}`:
  - `.attest/artifacts/analyze_results/current_analysis_plan.json`
  - If it does not exist, read `.attest/artifacts/analyze_results/current_analysis.md` instead
- Read `.attest/artifacts/execute_tests/current_execution_log.txt` only if you need to confirm failure details.
- Use `inspect_python` as needed to obtain the target signature, annotations, docstring, and source excerpt.
  - If the target file already exists, **do not overwrite the entire file with `write_file`**. Only block-level replacement is allowed.

## Specification (`test_plan.json`, if present)
{test_plan_json}

## Generation Rules (Follow the Specification)
1) **First round** (`epoch=1` and no analysis plan): generate only `SMOKE_SET` and use only weak assertions; keep `DEFERRED_SET` as placeholders.
2) **Later rounds** (analysis plan exists): modify only the listed BLOCKs (1-3 of them). If there are no failures but deferred items remain, promote the highest-priority CASE.
3) **Low/Medium must not be standalone CASE blocks**: they may only extend High-priority CASE blocks through parameter dimensions (`param_extensions`).
4) **Assertion levels**: enable strong assertions only when the specification explicitly allows them (for example, in the final or strong-assertion round).
5) **Strict budget enforcement**: follow `size`, `max_lines`, `max_params`, `is_parametrized`, and `requires_mock`. Split or reduce combinations if limits are exceeded.
6) **Module grouping**: if `groups` exist, process only the CASE blocks for `active_group`; defer the other groups.
7) **Test file path**: if the specification provides grouped paths in `test_files.groups`, use them; otherwise use `{test_file_path}`.

## BLOCK Index (If Present)
{block_index}

## Output
- Write `{test_file_path}` using a skeleton-first, then block-fill approach (relative path under `{project_root}`).
  - Step 1: **Only if the target file does not exist**, use `write_file` to create a compact skeleton (preferably under 200 lines) containing imports, fixed helpers/fixtures, test class/function declarations, and BLOCK placeholders marked with START/END. Placeholders should cover `SMOKE_SET` and `DEFERRED_SET`.
  - Step 2: Fill blocks in order (`HEADER` -> `CASE_*` -> `FOOTER`), preferably using `replace_block` once per block.
  - Step 3: Use `read_file` or `part_read` only when block location fails, and limit reads for that block to a single attempt.
  - Do not write the entire large file in one `write_file` call; do not clear or overwrite filled blocks; prefer fixing failing assertions only; add new cases only when necessary; never delete existing tests.
  - Medium/Low scenarios may only extend High-priority CASE blocks through parametrization and must not create standalone CASE blocks.

## Code Requirements
1. Use `pytest` with standard naming: `test_*.py` files and `test_*` functions.
2. Cover all test cases in `test_plan` and include the constraints from `requirements` such as shape, dtype, and exception behavior.
3. Fix random seeds when constructing inputs, avoid external resources, and use `unittest.mock` / `monkeypatch` stubs when external dependencies are unavoidable.
4. Assert return values, side effects, and exceptions explicitly; use suitable tolerances for floating-point comparisons.
5. If the target is a class or method, include instantiation logic or use a simplified fake implementation / fixture.
6. Keep the tests CPU-compatible and avoid heavy GPU or distributed dependencies unless `requirements` explicitly demand them.

## Suggested Structure
```python
import math
import pytest
from package.module import target  # Fill this in based on {target_fqn}

def test_happy_path():
    ...

@pytest.mark.parametrize(...)
def test_edge_case(...):
    ...

def test_invalid_inputs(...):
    with pytest.raises(...):
        ...
```

Add the required imports at the top of the file, including the target function or class, and keep the code directly runnable.
Write the file only through the multi-step tool flow above. Do not paste source code into the conversation.
"""

    def _build_block_index(self, path: Path) -> str:
        if not path.exists() or not path.is_file():
            return "N/A (test file not found)"
        return build_block_index_json(path)

    def _load_test_plan_json(self, state) -> str:
        plan_paths = [
            state.artifacts_dir / "design_test_plan" / "current_test_plan.json",
            Path(state.project_root) / "test_plan.json",
        ]
        for plan_path in plan_paths:
            if plan_path.exists() and plan_path.is_file():
                try:
                    return plan_path.read_text(encoding="utf-8")
                except Exception:
                    continue
        return "N/A (test_plan.json not found)"

    def _resolve_active_group(self, plan_text: str, epoch_current: int) -> str:
        if not plan_text or plan_text.startswith("N/A"):
            return "default"
        try:
            plan = json.loads(plan_text)
        except Exception:
            return "unknown"
        order = plan.get("active_group_order") or []
        if isinstance(order, list) and order:
            index = max(0, epoch_current - 1)
            if index < len(order):
                return str(order[index])
            return "all"
        return "default"

    def get_prompt_vars(
        self,
        state,
        target: str,
        target_slug: str,
        test_file_path: str,
        output_binary: str,
    ):
        path = Path(state.project_root) / test_file_path
        plan_text = self._load_test_plan_json(state)
        analysis_plan_path = state.artifacts_dir / "analyze_results" / "current_analysis_plan.json"
        analysis_plan_status = "present" if analysis_plan_path.exists() else "missing"
        epoch_current = getattr(state, "epoch_current", 1)
        epoch_total = getattr(state, "epoch_total", 1)
        active_group_hint = self._resolve_active_group(plan_text, epoch_current)
        return {
            "block_index": self._build_block_index(path),
            "test_plan_json": plan_text,
            "analysis_plan_status": analysis_plan_status,
            "epoch_current": epoch_current,
            "epoch_total": epoch_total,
            "active_group_hint": active_group_hint,
        }
    
    def get_config(self) -> StageConfig:
        return self.config
