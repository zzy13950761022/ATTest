"""
Stage 6: Analyze Results
Analyzes test execution logs to identify issues and suggest fixes.
"""
from ..stage import Stage, StageConfig


class AnalysisStage(Stage):
    """
    Analyze test execution results, identify failures, and suggest fixes.
    """
    
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="analyze_results",
            display_name="Analyze Results",
            description="Parse pytest logs and identify issues",
            prompt_template=self._get_prompt_template(),
            input_artifacts=["exit_code.txt"],
            output_artifacts=["analysis.md", "analysis_plan.json"],
            tools=["write_file", "read_file", "part_read", "list_files", "search"],
            allow_skip=False
        )
    
    def _get_prompt_template(self) -> str:
        return """You are the test analysis assistant. Interpret the pytest execution log and produce a block-level remediation plan.

Stage: Stage 6 - Analyze Results -> Output `analysis.md` and `analysis_plan.json`

## Inputs
- Execution log path: `.attest/artifacts/execute_tests/current_execution_log.txt`
- exit code：{exit_code_txt}

## Reading Strategy (Must Follow)
1) **Do not** read the entire log into the prompt directly.
2) First use `search` to locate key fragments, such as `FAILED`, `ERROR`, `AssertionError`, or `Traceback`.
3) Then use `part_read` to read the relevant context and take only the necessary fragments.
4) Use `read_file` for the full log only when the log is very short.

## Output Requirements (Must Follow)
1) **Output only a block-level remediation plan** and avoid long-form analysis.
2) In each round, list at most 1-3 BLOCKs to modify; mark the rest as deferred.
3) Every failing test must map to a `BLOCK_ID`. Prefer the `block_id` values in `test_plan.json`; use `HEADER` for shared dependencies/imports/fixtures and `FOOTER` for cleanup/tail logic.
4) If you find coverage gaps, you may add a new BLOCK (`action=add_case`), but still stay within the 1-3 BLOCK limit.
5) If the current failing set is identical to the previous round, you may set `stop_recommended=true` and provide a `stop_reason`. If only the error type repeats, do not stop; instead mark the BLOCK as deferred and note that the error type repeated and the block is being skipped.

## `analysis_plan.json` (Machine Readable)
Write strict JSON with the following fields:
```
{{
  "status": "success|not_fully_passed|failed",
  "passed": <int>,
  "failed": <int>,
  "errors": <int>,
  "collection_errors": <bool>,
  "block_limit": 3,
  "failures": [
    {{
      "test": "<node id>",
      "block_id": "<BLOCK_ID>",
      "error_type": "<AssertionError/TypeError/...>",
      "action": "rewrite_block|adjust_assertion|fix_dependency|add_case|mark_xfail",
      "note": "<short reason>"
    }}
  ],
  "deferred": [
    {{"test": "<node id>", "reason": "<short>"}}
  ],
  "stop_recommended": <bool>,
  "stop_reason": "<short>"
}}
```

## `analysis.md` (Concise and Readable)
Include only:
- Status plus passed / failed statistics
- A list of BLOCKs to fix (<=3, including `action` and `error_type`)
- `stop_recommended` / `stop_reason` when stopping is recommended

Use `write_file` to save `analysis_plan.json` and `analysis.md`. In the conversation, provide only a one-sentence summary."""
    
    def get_config(self) -> StageConfig:
        return self.config
