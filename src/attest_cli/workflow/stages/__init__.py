"""
Workflow stages (sub-agents) for test generation.
"""
from .understand import UnderstandFunctionStage
from .requirements import RequirementsStage
from .planning import TestPlanStage
from .codegen import CodeGenStage
from .execution import ExecutionStage
from .analysis import AnalysisStage
from .report import ReportStage


def build_all_stages(llm, tool_runner):
    """
    Build all workflow stages.
    
    Args:
        llm: LLM client instance
        tool_runner: Tool runner instance
    
    Returns:
        Dictionary mapping stage names to Stage instances
    """
    return {
        "understand_function": UnderstandFunctionStage(llm, tool_runner),
        "generate_requirements": RequirementsStage(llm, tool_runner),
        "design_test_plan": TestPlanStage(llm, tool_runner),
        "generate_code": CodeGenStage(llm, tool_runner),
        "execute_tests": ExecutionStage(llm, tool_runner),
        "analyze_results": AnalysisStage(llm, tool_runner),
        "generate_report": ReportStage(llm, tool_runner),
    }


__all__ = [
    "UnderstandFunctionStage",
    "RequirementsStage",
    "TestPlanStage",
    "CodeGenStage",
    "ExecutionStage",
    "AnalysisStage",
    "ReportStage",
    "build_all_stages",
]
