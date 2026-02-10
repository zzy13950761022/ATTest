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
        return """你是报告生成助手，请基于以下产物汇总 Python 目标 `{target_fqn}` 的测试结果。

## 可用材料
- 函数说明: `function_doc.md`
- 需求: `requirements.md`
- 测试计划: `test_plan.md`
- 结果分析: `analysis.md`
- 自动停止原因（如有）: {auto_stop_reason}
- 用户反馈: {user_feedback}

## 输出：`final_report.md`（使用 `write_file` 保存）
仅允许写入 `final_report.md`，不要覆盖/修改其他产物；如需查看产物内容请使用 `read_file`。
推荐结构：
1. 执行摘要：一句话结论 + 关键发现/阻塞项。
2. 测试范围：目标 FQN、环境（pytest + 依赖）、覆盖的场景/未覆盖项。
3. 结果概览：用例总数，通过/失败/错误数量，主要失败点。
4. 详细发现：按严重级别列出问题、根因、建议修复动作。
5. 覆盖与风险：需求覆盖、尚未覆盖的边界/缺失信息。
6. 后续动作：优先级排序的 TODO（修复测试/补充用例/环境调整）。

保持简洁、结构化，可供研发快速落地。不要在对话中粘贴全文，只写入文件。

## 参考内容（来自上一阶段产物）
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
        reason = getattr(state, "auto_stop_reason", "") or "无"
        return {"auto_stop_reason": reason}

    def get_config(self) -> StageConfig:
        return self.config
