"""
Workflow profile registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


DEFAULT_STAGE_NAMES = [
    "understand_function",
    "generate_requirements",
    "design_test_plan",
    "validate_infrastructure",
    "generate_code",
    "execute_tests",
    "analyze_results",
    "generate_report",
]


@dataclass(frozen=True)
class WorkflowProfile:
    name: str
    display_name: str
    stage_names: List[str]
    build_stages: Callable


def get_workflow_profiles() -> Dict[str, WorkflowProfile]:
    from .stages import build_all_stages
    from .ascend_stages import build_ascend_stages, build_ascend_continuous_stages

    return {
        "python": WorkflowProfile(
            name="python",
            display_name="Python Workflow",
            stage_names=list(DEFAULT_STAGE_NAMES),
            build_stages=build_all_stages,
        ),
        "ascend_ut": WorkflowProfile(
            name="ascend_ut",
            display_name="Ascend UT Workflow",
            stage_names=list(DEFAULT_STAGE_NAMES),
            build_stages=build_ascend_stages,
        ),
        "ascend_ut_continuous": WorkflowProfile(
            name="ascend_ut_continuous",
            display_name="Ascend UT Continuous Agent",
            stage_names=[
                "understand_function",
                "generate_requirements",
                "design_test_plan",
                "validate_infrastructure",
                "generate_code",
                "generate_report",
            ],
            build_stages=build_ascend_continuous_stages,
        ),
    }


def get_workflow_profile(name: str) -> WorkflowProfile:
    profiles = get_workflow_profiles()
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown workflow profile '{name}'. Available: {available}")
    return profiles[name]
