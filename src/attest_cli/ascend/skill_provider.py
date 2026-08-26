"""
Host-side progressive disclosure for Ascend UT skill references.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict


DEFAULT_ASCEND_SKILL_ROOT = "/mnt/fangcr/ops-math-round4/.claude/skills/ascendc-ut-develop"


@dataclass
class SkillDoc:
    name: str
    text: str


class AscendSkillProvider:
    def __init__(self, skill_root: str | Path = DEFAULT_ASCEND_SKILL_ROOT):
        self.skill_root = Path(skill_root).expanduser().resolve()
        self.docs = self._load_docs()

    def _load_docs(self) -> Dict[str, SkillDoc]:
        rel_paths = {
            "skill": "SKILL.md",
            "workflow": "references/ut-generator/ut-generator-workflow.md",
            "op_host": "references/ut-generator/op-host-ut-generator.md",
            "op_api": "references/ut-generator/op-api-ut-generator.md",
            "op_kernel": "references/ut-generator/op-kernel-ut-generator.md",
            "ce_workflow": "references/coverage-enhancement/coverage-enhancement-workflow.md",
            "ce_baseline": "references/coverage-enhancement/baseline-coverage-agent.md",
            "ce_impl": "references/coverage-enhancement/test-implementation-agent.md",
            "ce_extract": "references/coverage-enhancement/coverage-extraction-guide.md",
        }
        docs: Dict[str, SkillDoc] = {}
        for key, rel_path in rel_paths.items():
            path = self.skill_root / rel_path
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                text = ""
            docs[key] = SkillDoc(name=key, text=text)
        return docs

    def _extract_heading(self, text: str, heading: str) -> str:
        if not text.strip():
            return ""
        pattern = re.compile(
            rf"(?ms)^##\s+{re.escape(heading)}.*?(?=^##\s+|\Z)"
        )
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
        return ""

    def _clip(self, text: str, limit: int = 1800) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "\n... (truncated)"

    def _join_nonempty(self, parts: list[str], limit: int = 1800) -> str:
        return self._clip("\n\n".join(part for part in parts if part), limit=limit)

    def _generate_packet(self, stage_name: str, layer_id: str | None = None) -> str:
        if stage_name == "understand_function":
            parts = [
                self._extract_heading(self.docs["skill"].text, "第一步：模式选择"),
                self._extract_heading(self.docs["workflow"].text, "Phase UT-1: 信息收集与自动探索"),
            ]
            return self._join_nonempty(parts)

        if stage_name in {"generate_requirements", "design_test_plan"}:
            parts = [
                self._extract_heading(self.docs["workflow"].text, "Phase UT-2: op_host UT 编写（P0 优先）"),
                self._extract_heading(self.docs["workflow"].text, "Phase UT-3: op_api UT 编写（P1 按需）"),
                self._extract_heading(self.docs["workflow"].text, "Phase UT-4: op_kernel UT 编写（P2 按需）"),
            ]
            return self._join_nonempty(parts, limit=2200)

        if stage_name == "generate_code":
            mapping = {
                "op_host": self.docs["op_host"].text,
                "op_api": self.docs["op_api"].text,
                "op_kernel": self.docs["op_kernel"].text,
                "op_kernel_aicpu": self.docs["op_kernel"].text,
            }
            text = mapping.get(layer_id or "", self.docs["workflow"].text)
            limit = 4000 if layer_id == "op_host" else 2500
            return self._clip(text, limit=limit)

        if stage_name == "generate_report":
            parts = [
                self._extract_heading(self.docs["skill"].text, "核心理念"),
                self._extract_heading(self.docs["workflow"].text, "Phase UT-6: 生成最终报告"),
            ]
            return self._join_nonempty(parts)

        return self._clip(self.docs["skill"].text)

    def _enhance_packet(self, stage_name: str, layer_id: str | None = None) -> str:
        if stage_name == "understand_function":
            parts = [
                self._extract_heading(self.docs["skill"].text, "第一步：模式选择"),
                self._extract_heading(self.docs["ce_workflow"].text, "适用场景"),
                self._extract_heading(self.docs["ce_workflow"].text, "Phase CE-1: Interview 模式"),
                self._extract_heading(self.docs["ce_workflow"].text, "Phase CE-2: 获取初始覆盖率"),
                self._extract_heading(self.docs["ce_extract"].text, "覆盖率目标"),
            ]
            return self._join_nonempty(parts, limit=2200)

        if stage_name in {"generate_requirements", "design_test_plan"}:
            parts = [
                self._extract_heading(self.docs["ce_workflow"].text, "Phase CE-2: 获取初始覆盖率"),
                self._extract_heading(self.docs["ce_workflow"].text, "Phase CE-3: 分析与实现测试用例"),
                self._extract_heading(self.docs["ce_workflow"].text, "Phase CE-4: 持续迭代优化"),
                self._extract_heading(self.docs["ce_extract"].text, "覆盖率目标"),
                self._extract_heading(self.docs["ce_extract"].text, "覆盖率提升策略"),
            ]
            return self._join_nonempty(parts, limit=2600)

        if stage_name == "generate_code":
            mapping = {
                "op_host": self.docs["op_host"].text,
                "op_api": self.docs["op_api"].text,
                "op_kernel": self.docs["op_kernel"].text,
                "op_kernel_aicpu": self.docs["op_kernel"].text,
            }
            parts = [
                self._extract_heading(self.docs["ce_workflow"].text, "Phase CE-3: 分析与实现测试用例"),
                self.docs["ce_impl"].text,
                self._extract_heading(self.docs["ce_extract"].text, "覆盖率提升策略"),
                mapping.get(layer_id or "", ""),
            ]
            return self._join_nonempty(parts, limit=8000)

        if stage_name == "generate_report":
            parts = [
                self._extract_heading(self.docs["skill"].text, "核心理念"),
                self._extract_heading(self.docs["ce_workflow"].text, "Phase CE-5: 生成最终报告"),
                self._extract_heading(self.docs["ce_workflow"].text, "Phase CE-6: 输出进入条件清单"),
                self._extract_heading(self.docs["ce_extract"].text, "覆盖率目标"),
            ]
            return self._join_nonempty(parts, limit=2200)

        return self._clip(self.docs["skill"].text)

    def get_stage_packet(
        self,
        stage_name: str,
        layer_id: str | None = None,
        generation_mode: str = "ut_generate",
    ) -> str:
        if generation_mode == "ut_enhance":
            return self._enhance_packet(stage_name, layer_id)
        return self._generate_packet(stage_name, layer_id)
