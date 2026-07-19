"""
Stage 7: Generate Report
**IMPORTANT**: Use the `write_file` tool to save your final report to "final_report.md".
Do not output the full document in your response..
"""
from ..stage import Stage, StageConfig


class ReportStage(Stage):
    """
    Generate final comprehensive test report.
    """
    
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="generate_report",
            display_name="Generate Report",
            description="Produce final test report",
            prompt_template=self._get_prompt_template(),
            input_artifacts=[
                "function_doc.md",
                "requirements.md",
                "test_plan.md",
                "analysis.md"
            ],
            output_artifacts=["final_report.md"],
            tools=["read_file", "write_file"],
            allow_skip=False
        )
    
    def _get_prompt_template(self) -> str:
        return """You are the report generation assistant. Summarize the test results for the Python target `{target_fqn}` based on the artifacts below.

## Available Materials
- Function documentation: `function_doc.md`
- Requirements: `requirements.md`
- Test plan: `test_plan.md`
- Result analysis: `analysis.md`
- Automatic stop reason (if any): {auto_stop_reason}
- User feedback: {user_feedback}

## Output: `final_report.md` (save it using `write_file`)
Only write to `final_report.md`. Do not overwrite or modify other artifacts. Use `read_file` if you need to inspect them.
Recommended structure:
1. Executive summary: a one-sentence conclusion plus key findings and blockers.
2. Test scope: target FQN, environment (pytest and dependencies), covered scenarios, and uncovered items.
3. Result overview: total number of cases, passed/failed/error counts, and the primary failure points.
4. Detailed findings: list issues, root causes, and suggested fixes by severity.
5. Coverage and risks: requirement coverage plus uncovered boundaries and missing information.
6. Next actions: priority-ordered TODOs for test fixes, new cases, or environment adjustments.

Keep the report concise and structured so engineers can act on it quickly. Do not paste the full report into the chat; write it only to the file.

## Reference Content (from the previous stage artifacts)
### function_doc.md
{function_doc_md}

### requirements.md
{requirements_md}

### test_plan.md
{test_plan_md}

### analysis.md
{analysis_md}
"""
    
    def get_prompt_vars(self, state, target, target_slug, test_file_path, output_binary):
        reason = getattr(state, "auto_stop_reason", "") or "none"
        return {"auto_stop_reason": reason}

    def get_config(self) -> StageConfig:
        return self.config
