"""
Stage 4: Generate Code
Creates test code files and build/run scripts.
"""
import json
import re
from pathlib import Path

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
        return """你正在为 Python 目标 `{target_fqn}` 生成 pytest 单测代码。

当前阶段：Stage 4 - 生成测试代码 → 目标文件 `{test_file_path}`

## 迭代状态
- epoch: {epoch_current}/{epoch_total}
- analysis_plan: {analysis_plan_status}
- active_group: {active_group_hint}

## 核心约束（必须遵守）
1) 语义分块：BLOCK 类型固定为 `HEADER` / `CASE_*` / `FOOTER`，`CASE` 表示单个用例或一个参数化组。
   - 标记格式：
     - `# ==== BLOCK:HEADER START ====`
     - `# ==== BLOCK:HEADER END ====`
     - `# ==== BLOCK:CASE_01 START ====`
     - `# ==== BLOCK:CASE_01 END ====`
     - `# ==== BLOCK:FOOTER START ====`
     - `# ==== BLOCK:FOOTER END ====`
   - BLOCK_ID 必须稳定，禁止重命名或重排已有 BLOCK。
2) 编辑次数限制：对任一 BLOCK_ID，最多 1 次读取 + 1 次替换。
   - 优先使用 `replace_block`，避免反复 `read_file`/`part_read`/`search`。
   - 若一次替换后仍失败，交给 analyze_results 决策是否重写。
3) 单次写入限制：每次 `write_file`/`replace_block` 内容 ≤ 8KB。
   - 超限时拆分为多个 CASE（例如 CASE_03A/CASE_03B），或使用参数化拆小块。
4) 增量迭代：仅修改分析计划中标记的问题 BLOCK（每轮 1~3 个）。
5) 禁止自循环：完成本轮替换后停止，不要在 generate_code 内多轮反复生成。

## 输入
- 必需：`requirements.md`、`test_plan.md`
- **优先规格书**：如有 `test_plan.json`，以其为唯一生成依据；无则从 `test_plan.md` 推导同等结构。
- 推荐：先用 `read_file` 读取 `function_doc.md`/`requirements.md`/`test_plan.md` 理解约束。
- 迭代修复：优先读取分析计划（路径相对 {project_root}）：
  - `.testagent/artifacts/analyze_results/current_analysis_plan.json`
  - 若不存在再读 `.testagent/artifacts/analyze_results/current_analysis.md`
- 如需核对失败详情再读执行日志：`.testagent/artifacts/execute_tests/current_execution_log.txt`
- 可按需 `inspect_python` 目标获取签名、注解、docstring、源码片段。
  - 如果目标文件已存在：**禁止使用 `write_file` 覆盖全文件**，仅允许按块替换。

## 规格书（test_plan.json，如存在）
{test_plan_json}

## 生成规则（按规格执行）
1) **首轮**（epoch=1 且无 analysis_plan）：只生成 `SMOKE_SET`，只用 weak 断言；`DEFERRED_SET` 只保留占位。
2) **后续轮**（analysis_plan 存在）：仅修改计划中列出的 BLOCK（1-3 个）；如无失败且 deferred 仍有，用优先级最高的 1 个 CASE 进入。
3) **Low/Medium 禁止独立 CASE**：只能作为 High CASE 的参数维度扩展（param_extensions）。
4) **断言分级**：只有当规格允许（final 或强断言轮）才启用 strong 断言。
5) **预算严格执行**：遵守 size/max_lines/max_params/is_parametrized/requires_mock；超限必须拆分或减少组合。
6) **模块分组**：若有 groups，只处理 active_group 对应的 CASE；其他 group 延后。
7) **测试文件路径**：若规格给出 group 文件（test_files.groups），使用对应路径；否则用 `{test_file_path}`。

## BLOCK 索引（若存在）
{block_index}

## 输出
- 采用 “先骨架、后分块填充” 写入 `{test_file_path}`（相对路径，位于 {project_root} 下）。
  - 第 1 步：**仅当目标文件不存在时**，用 `write_file` 写入精简骨架（行数尽量 < 200），只包含 import、固定 helper/fixture、测试类/函数声明和 BLOCK 占位段（使用 START/END 标记）。占位段覆盖：`SMOKE_SET` + `DEFERRED_SET`（仅占位）。
  - 第 2 步：按块顺序填充（HEADER → CASE_* → FOOTER），优先使用 `replace_block` 一次性写入每个块内容。
  - 第 3 步：只有当块定位失败时才 `read_file`/`part_read` 辅助定位，并限制对该块的读取次数为 1 次。
  - 禁止在一次 `write_file` 中写入完整大文件；禁止清空/覆盖已填充块；优先只修复失败断言，必要时新增用例，禁止删除已有测试。
  - Medium/Low 场景只能通过参数化扩展 High CASE，不得新增独立 CASE。

## 代码要求
1. 使用 `pytest`，命名规范：`test_*.py`、`test_*` 函数。
2. 覆盖 test_plan 中的所有测试用例，补充 requirements 里的约束（shape/dtype/异常）。
3. 构造输入时固定随机种子，避免依赖外部资源；如需外部依赖，使用 `unittest.mock`/`monkeypatch` stub。
4. 对返回值/副作用/异常做明确断言；浮点比较使用合适的容差。
5. 若目标是类/方法，包含实例化逻辑或使用简化的假实现/fixture。
6. 兼容 CPU 环境，避免 GPU/分布式等重度依赖，除非需求明确。

## 建议结构
```python
import math
import pytest
from package.module import target  # 根据 {target_fqn} 填写

def test_happy_path():
    ...

@pytest.mark.parametrize(...)
def test_edge_case(...):
    ...

def test_invalid_inputs(...):
    with pytest.raises(...):
        ...
```

在文件头部添加必要的 import（含目标函数/类），保持代码可直接运行。
只通过上述多步工具写入文件，不要在对话中粘贴源代码。
"""

    def _build_block_index(self, path: Path) -> str:
        if not path.exists() or not path.is_file():
            return "N/A (test file not found)"
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return "N/A (unable to read test file)"

        pattern = re.compile(
            r"^#\s*====\s*BLOCK:(?P<block_id>[A-Za-z0-9_]+)(?P<suffix>\s+(START|END))?\s*====\s*$"
        )
        blocks = {}
        for line_num, line in enumerate(text.splitlines(), 1):
            match = pattern.match(line)
            if not match:
                continue
            block_id = match.group("block_id")
            suffix = match.group("suffix") or ""
            entry = blocks.setdefault(
                block_id,
                {"block_id": block_id, "start_line": None, "end_line": None, "markers": []},
            )
            entry["markers"].append(line_num)
            if "START" in suffix:
                entry["start_line"] = line_num
            elif "END" in suffix:
                entry["end_line"] = line_num
            else:
                if entry["start_line"] is None:
                    entry["start_line"] = line_num
                if entry["end_line"] is None:
                    entry["end_line"] = line_num

        if not blocks:
            return "N/A (no block markers found)"

        entries = []
        for entry in blocks.values():
            start = entry["start_line"]
            end = entry["end_line"]
            if start and end and start == end:
                status = "placeholder"
            elif start and end:
                status = "bounded"
            else:
                status = "open"
            entries.append(
                {
                    "block_id": entry["block_id"],
                    "start_line": start,
                    "end_line": end,
                    "status": status,
                }
            )

        entries.sort(key=lambda item: item["start_line"] or 0)
        if len(entries) > 120:
            entries = entries[:120]
            entries.append({"truncated": True, "note": "too many blocks"})

        return json.dumps(entries, ensure_ascii=True, indent=2)

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
