import json
import sys
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent / "src"))

from attest_cli.ascend import inspect_ascend_operator
from attest_cli.ascend.skill_provider import AscendSkillProvider
from attest_cli.block_utils import build_block_entries
from attest_cli.cli import app
from attest_cli.llm import ChatResponse
from attest_cli.tools import ToolContext
from attest_cli.tools.base import Tool
from attest_cli.tools.builtin import ReadFileTool, ReplaceBlockTool, SearchTool
from attest_cli.tools.runner import ToolRegistry, ToolRunner
from attest_cli.workflow import WorkflowEngine


SKILL_ROOT = "/Users/zzf1sh/Documents/Project/ops-math/.claude/skills/ascendc-ut-develop"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_skill_root(base: Path) -> Path:
    skill_root = base / "skill-root"
    _write(
        skill_root / "SKILL.md",
        "# Skill\n\n## 第一步：模式选择\nGEN_MODE\n\n## 核心理念\nCORE_IDEA\n",
    )
    _write(
        skill_root / "references" / "ut-generator" / "ut-generator-workflow.md",
        "# Workflow\n\n## Phase UT-1: 信息收集与自动探索\nUT1\n\n## Phase UT-2: op_host UT 编写（P0 优先）\nUT2\n\n## Phase UT-3: op_api UT 编写（P1 按需）\nUT3\n\n## Phase UT-4: op_kernel UT 编写（P2 按需）\nUT4\n\n## Phase UT-6: 生成最终报告\nUT6\n",
    )
    _write(skill_root / "references" / "ut-generator" / "op-host-ut-generator.md", "OP_HOST_GUIDE\n")
    _write(skill_root / "references" / "ut-generator" / "op-api-ut-generator.md", "OP_API_GUIDE\n")
    _write(skill_root / "references" / "ut-generator" / "op-kernel-ut-generator.md", "OP_KERNEL_GUIDE\n")
    _write(
        skill_root / "references" / "coverage-enhancement" / "coverage-enhancement-workflow.md",
        "# Coverage Enhancement\n\n## 适用场景\nCE_SCENE\n\n## Phase CE-1: Interview 模式\nCE1\n\n## Phase CE-2: 获取初始覆盖率\nCE2\n\n## Phase CE-3: 分析与实现测试用例\nCE3\n\n## Phase CE-4: 持续迭代优化\nCE4\n\n## Phase CE-5: 生成最终报告\nCE5\n\n## Phase CE-6: 输出进入条件清单\nCE6\n",
    )
    _write(skill_root / "references" / "coverage-enhancement" / "baseline-coverage-agent.md", "BASELINE_AGENT\n")
    _write(skill_root / "references" / "coverage-enhancement" / "test-implementation-agent.md", "IMPL_AGENT\n")
    _write(
        skill_root / "references" / "coverage-enhancement" / "coverage-extraction-guide.md",
        "# Coverage Extraction\n\n## 覆盖率目标\nCOV_TARGET\n\n## 覆盖率提升策略\nCOV_STRATEGY\n",
    )
    return skill_root


def _make_repo(base: Path, *, with_op_api: bool = False, with_existing_ut: bool = False) -> Path:
    repo_root = base / "ops-math"
    build_sh = """#!/usr/bin/env bash
if [ "$1" = "-h" ]; then
  echo "--ophost --opapi --opkernel --soc Ascend910B Ascend950 --cov"
  exit 0
fi

layer=""
if printf '%s' "$*" | grep -q -- "--ophost"; then
  layer="op_host"
elif printf '%s' "$*" | grep -q -- "--opapi"; then
  layer="op_api"
fi

echo "BUILD_OK $*"

if printf '%s' "$*" | grep -q -- "--cov"; then
  target_dir=""
  baseline_cov="0.0"
  generated_cov="0.0"
  if [ "$layer" = "op_host" ]; then
    target_dir="math/add/tests/ut/op_host"
    baseline_cov="70.0"
    generated_cov="86.5"
  elif [ "$layer" = "op_api" ]; then
    target_dir="math/add/tests/ut/op_api"
    baseline_cov="61.0"
    generated_cov="84.0"
  fi

  if [ -n "$target_dir" ] && [ -d "$target_dir" ] && grep -R -q "CASE_" "$target_dir"; then
    echo "COVERAGE_LAYER ${layer} ${generated_cov}"
  else
    echo "COVERAGE_LAYER ${layer} ${baseline_cov}"
  fi
fi

exit 0
"""
    _write(repo_root / "build.sh", build_sh)
    _write(repo_root / "tests" / "ut" / "common" / "CMakeLists.txt", "add_library(common_ut OBJECT empty.cpp)\n")
    _write(repo_root / "tests" / "ut" / "op_host" / "CMakeLists.txt", "add_executable(op_host_ut test_op_host_main.cpp)\n")
    _write(repo_root / "tests" / "ut" / "op_host" / "test_op_host_main.cpp", "int main() { return 0; }\n")
    _write(repo_root / "common" / "inc" / "op_host" / "tiling_base.h", "struct DummyTilingBase {};\n")
    _write(repo_root / "common" / "inc" / "op_host" / "tiling_templates_registry.h", "struct DummyRegistry {};\n")
    _write(repo_root / "common" / "inc" / "op_host" / "tiling_util.h", "inline int DummyTilingUtil() { return 0; }\n")
    _write(repo_root / "common" / "inc" / "op_host" / "input_util.h", "inline int DummyInputUtil() { return 0; }\n")
    if with_op_api:
        _write(repo_root / "tests" / "ut" / "op_api" / "CMakeLists.txt", "add_executable(op_api_ut test_op_api_main.cpp)\n")
        _write(repo_root / "tests" / "ut" / "op_api" / "test_op_api_main.cpp", "int main() { return 0; }\n")

    tiling_cpp = """
IMPL_OP_OPTILING(Add).TilingParse<BroadcastCompileInfo>(context, compileInfo);
auto dtype = ge::DT_FLOAT16;
auto format = ge::FORMAT_ND;
context->GetAttr("adj_x1");
context->GetAttrValue("adj_x2");
"""
    _write(repo_root / "math" / "add" / "op_host" / "arch35" / "add_tiling_arch35.cpp", tiling_cpp)
    _write(repo_root / "math" / "add" / "op_host" / "add.cpp", "ge::DT_FLOAT16 ge::FORMAT_ND\n")
    if with_op_api:
        _write(repo_root / "math" / "add" / "op_api" / "aclnn_add.cpp", "return ACL_SUCCESS;\n")
    if with_existing_ut:
        _write(
            repo_root / "math" / "add" / "tests" / "ut" / "op_host" / "CMakeLists.txt",
            "add_library(existing_add_host OBJECT test_add_tiling.cpp)\n",
        )
        _write(
            repo_root / "math" / "add" / "tests" / "ut" / "op_host" / "test_add_tiling.cpp",
            "// existing host baseline\nTEST(ExistingHost, Baseline) { EXPECT_TRUE(true); }\n",
        )
        if with_op_api:
            _write(
                repo_root / "math" / "add" / "tests" / "ut" / "op_api" / "CMakeLists.txt",
                "add_library(existing_add_api OBJECT test_aclnn_add.cpp)\n",
            )
            _write(
                repo_root / "math" / "add" / "tests" / "ut" / "op_api" / "test_aclnn_add.cpp",
                "// existing api baseline\nTEST(ExistingApi, Baseline) { EXPECT_TRUE(true); }\n",
            )
    return repo_root


def _make_engine(repo_root: Path, workspace: Path, *, epochs: int = 1) -> WorkflowEngine:
    return WorkflowEngine(
        llm=MockAscendLLM(),
        workspace=str(workspace),
        op="add",
        arch="ascendc",
        soc="Ascend910B",
        vendor="ascend",
        project_root=str(repo_root),
        target="ops-math/math/add",
        workflow_kind="ascend_ut",
        op_path="ops-math/math/add",
        repo_name="ops-math",
        category="math",
        op_name="add",
        skill_root=SKILL_ROOT,
        epochs=epochs,
    )


class MockAscendLLM:
    def __init__(self):
        self._seen = {}

    def chat(self, messages, tools=None):
        prompt = messages[-1]["content"] if messages else ""
        if "Target file:" not in prompt:
            return ChatResponse("noop")

        if self._seen.get(prompt):
            return ChatResponse("done")
        self._seen[prompt] = True

        file_match = next(
            (line.split("`")[1] for line in prompt.splitlines() if line.startswith("Target file:")),
            "",
        )
        block_match = next(
            (line.split(":", 1)[1].strip() for line in prompt.splitlines() if line.startswith("Target blocks to fill or revise:")),
            "",
        )
        block_ids = [item.strip() for item in block_match.split(",") if item.strip()]
        is_cmake = file_match.endswith("CMakeLists.txt")

        tool_calls = []
        for index, block_id in enumerate(block_ids):
            if block_id == "HEADER":
                if is_cmake:
                    content = "cmake_minimum_required(VERSION 3.16)\nset(OP_TEST_TARGET add_ut)"
                else:
                    content = '#include "gtest/gtest.h"\nclass AddTest : public testing::Test {};'
            elif block_id == "FOOTER":
                content = "# end" if is_cmake else "// end"
            else:
                if is_cmake:
                    source_file = "test_add_tiling.cpp" if "op_host" in file_match else "test_aclnn_add.cpp"
                    content = f"add_library({block_id.lower()} OBJECT {source_file})"
                else:
                    content = (
                        f"TEST_F(AddTest, {block_id}_Smoke_{index}) {{\n"
                        f"    EXPECT_TRUE(true);\n"
                        f"}}"
                    )
            tool_calls.append(
                {
                    "id": f"tc_{index}",
                    "type": "function",
                    "function": {
                        "name": "replace_block",
                        "arguments": json.dumps(
                            {
                                "path": file_match,
                                "block_id": block_id,
                                "content": content,
                            }
                        ),
                    },
                }
            )
        return ChatResponse("", tool_calls=tool_calls)


class RecoveringToolErrorLLM:
    def __init__(self, bad_absolute_path: str):
        self.bad_absolute_path = bad_absolute_path
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(
                "",
                tool_calls=[
                    {
                        "id": "bad_call",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {
                                    "path": self.bad_absolute_path,
                                    "content": "bad",
                                }
                            ),
                        },
                    }
                ],
            )
        if self.calls == 2:
            return ChatResponse(
                "",
                tool_calls=[
                    {
                        "id": "good_call",
                        "type": "function",
                        "function": {
                            "name": "replace_block",
                            "arguments": json.dumps(
                                {
                                    "path": "demo.cpp",
                                    "block_id": "HEADER",
                                    "content": "int generated = 1;",
                                }
                            ),
                        },
                    }
                ],
            )
        return ChatResponse("done")


def test_inspect_ascend_operator_normalizes_and_extracts_context(tmp_path):
    repo_root = _make_repo(tmp_path, with_op_api=True)

    context = inspect_ascend_operator(repo_root, "math/add", soc_hint="Ascend910B")

    assert context["op_path"] == "ops-math/math/add"
    assert context["generation_mode"] == "ut_generate"
    assert context["coverage_mode"] == "single_run"
    assert context["enabled_layers"] == ["op_host", "op_api"]
    assert context["compare_scope"] == ["op_host", "op_api"]
    assert "DT_FLOAT16" in context["dtype_candidates"]
    assert "FORMAT_ND" in context["format_candidates"]
    assert "adj_x1" in context["required_attrs"]
    assert context["compile_info_candidates"][0]["type"] == "BroadcastCompileInfo"
    assert context["reference_context"]["shared_test_harness"]
    assert context["reference_context"]["shared_helpers"]


def test_inspect_ascend_operator_detects_existing_ut(tmp_path):
    repo_root = _make_repo(tmp_path, with_op_api=True, with_existing_ut=True)

    context = inspect_ascend_operator(repo_root, "ops-math/math/add", soc_hint="Ascend910B")

    assert context["generation_mode"] == "ut_enhance"
    assert context["coverage_mode"] == "before_after_compare"
    assert context["existing_ut_detected"] is True
    assert any("test_add_tiling.cpp" in path for path in context["existing_ut_by_layer"]["op_host"])
    assert any("test_aclnn_add.cpp" in path for path in context["existing_ut_by_layer"]["op_api"])


def test_skill_provider_routes_generate_and_enhance_packets(tmp_path):
    skill_root = _make_skill_root(tmp_path)
    provider = AscendSkillProvider(skill_root=skill_root)

    generate_understand = provider.get_stage_packet("understand_function", generation_mode="ut_generate")
    enhance_understand = provider.get_stage_packet("understand_function", generation_mode="ut_enhance")
    generate_code = provider.get_stage_packet("generate_code", "op_host", generation_mode="ut_generate")
    enhance_code = provider.get_stage_packet("generate_code", "op_host", generation_mode="ut_enhance")
    enhance_report = provider.get_stage_packet("generate_report", generation_mode="ut_enhance")

    assert "UT1" in generate_understand
    assert "CE1" in enhance_understand
    assert "CE2" in enhance_understand
    assert "OP_HOST_GUIDE" in generate_code
    assert "IMPL_AGENT" in enhance_code
    assert "COV_STRATEGY" in enhance_code
    assert "OP_HOST_GUIDE" in enhance_code
    assert "CE5" in enhance_report
    assert "CE6" in enhance_report


def test_replace_block_supports_hash_and_slash_markers(tmp_path):
    ctx = ToolContext(cwd=str(tmp_path), auto_approve=True)
    tool = ReplaceBlockTool()

    cpp = tmp_path / "test.cpp"
    cmake = tmp_path / "CMakeLists.txt"
    _write(cpp, "// ==== BLOCK:CASE_01 ====\n")
    _write(cmake, "# ==== BLOCK:HEADER ====\n")

    cpp_result = tool.execute({"path": "test.cpp", "block_id": "CASE_01", "content": "int x = 1;"}, ctx)
    cmake_result = tool.execute({"path": "CMakeLists.txt", "block_id": "HEADER", "content": "cmake_minimum_required(VERSION 3.16)"}, ctx)

    assert cpp_result.ok and cmake_result.ok
    assert "// ==== BLOCK:CASE_01 START ====" in cpp.read_text(encoding="utf-8")
    assert "# ==== BLOCK:HEADER START ====" in cmake.read_text(encoding="utf-8")


def test_read_file_returns_error_for_directory(tmp_path):
    target_dir = tmp_path / "math" / "add" / "op_host"
    target_dir.mkdir(parents=True, exist_ok=True)

    tool = ReadFileTool()
    ctx = ToolContext(cwd=str(tmp_path), auto_approve=True)
    result = tool.execute({"path": "math/add/op_host"}, ctx)

    assert not result.ok
    assert "directory" in (result.error or "").lower()


def test_search_supports_absolute_paths_outside_cwd(tmp_path):
    cwd = tmp_path / "generated_repo"
    external_root = tmp_path / "ops-math"
    cwd.mkdir(parents=True, exist_ok=True)
    target = external_root / "math" / "add" / "CMakeLists.txt"
    _write(target, "add_executable(add_ut main.cpp)\n")

    tool = SearchTool()
    ctx = ToolContext(cwd=str(cwd), auto_approve=True)
    result = tool.execute(
        {"pattern": "add_executable", "path": str(external_root), "file_pattern": "CMakeLists.txt"},
        ctx,
    )

    assert result.ok
    assert str(target) in result.output


def test_tool_runner_catches_tool_exception(tmp_path):
    class ExplodingTool(Tool):
        name = "explode"
        readonly = True
        description = "raise for testing"

        def execute(self, params, ctx):
            raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(ExplodingTool)
    runner = ToolRunner(registry)

    result = runner.execute("explode", {}, ToolContext(cwd=str(tmp_path), auto_approve=True))

    assert not result.ok
    assert "RuntimeError" in (result.error or "")


def test_codegen_session_recovers_after_invalid_absolute_write(tmp_path):
    repo_root = _make_repo(tmp_path)
    workspace = tmp_path / "workspace"
    engine = _make_engine(repo_root, workspace)
    stage = engine.stages["generate_code"]
    stage.llm = RecoveringToolErrorLLM("/mnt/fangcr/ops-math/math/add/tests/ut/op_host/test_add_infershape.cpp")

    generated_root = workspace / ".attest" / "generated_repo"
    generated_root.mkdir(parents=True, exist_ok=True)
    _write(generated_root / "demo.cpp", "// ==== BLOCK:HEADER ====\n")

    result = stage._run_llm_session(engine.state, "demo prompt", generated_root)

    assert result.success
    assert "generated = 1" in (generated_root / "demo.cpp").read_text(encoding="utf-8")


def test_ascend_cli_parses_and_initializes_engine(tmp_path, monkeypatch):
    repo_root = _make_repo(tmp_path)
    runner = CliRunner()
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, mode):
            captured["mode"] = mode

    monkeypatch.setattr("attest_cli.workflow.WorkflowEngine", FakeEngine)

    result = runner.invoke(
        app,
        [
            "ascend-ut",
            "run",
            "--project-root",
            str(repo_root),
            "--op-path",
            "math/add",
            "--workspace",
            str(tmp_path / "workspace"),
            "--soc",
            "Ascend910B",
        ],
    )

    assert result.exit_code == 0
    assert captured["workflow_kind"] == "ascend_ut"
    assert captured["op_path"] == "ops-math/math/add"
    assert captured["op_name"] == "add"
    assert captured["mode"] == "interactive"


def test_stage1_and_stage3_generate_ascend_artifacts(tmp_path):
    repo_root = _make_repo(tmp_path, with_op_api=True)
    workspace = tmp_path / "workspace"
    engine = _make_engine(repo_root, workspace)

    assert engine.stages["understand_function"].execute(engine.state).success
    assert engine.stages["generate_requirements"].execute(engine.state).success
    plan_result = engine.stages["design_test_plan"].execute(engine.state)

    assert plan_result.success
    plan = json.loads(engine.state.load_artifact("test_plan.json"))
    assert plan["workflow_kind"] == "ascend_ut"
    assert plan["generation_mode"] == "ut_generate"
    assert plan["coverage_mode"] == "single_run"
    assert plan["generated"]["isolated_snapshot"] is False
    assert any(item["kind"] == "cmake" for item in plan["files"])
    assert any(item["layer_id"] == "op_api" for item in plan["files"])
    assert plan["smoke_set"]


def test_stage1_uses_existing_ut_as_baseline(tmp_path):
    repo_root = _make_repo(tmp_path, with_op_api=True, with_existing_ut=True)
    workspace = tmp_path / "workspace"
    engine = _make_engine(repo_root, workspace)

    result = engine.stages["understand_function"].execute(engine.state)

    assert result.success
    assert result.complete_workflow is False
    assert engine.state.generation_mode == "ut_enhance"
    assert engine.state.coverage_mode == "before_after_compare"
    assert "enhance them in a snapshot" in result.message


def test_enhance_mode_stage4_generates_companion_files_in_snapshot(tmp_path):
    repo_root = _make_repo(tmp_path, with_op_api=True, with_existing_ut=True)
    workspace = tmp_path / "workspace"
    engine = _make_engine(repo_root, workspace)

    assert engine.stages["understand_function"].execute(engine.state).success
    assert engine.stages["generate_requirements"].execute(engine.state).success
    assert engine.stages["design_test_plan"].execute(engine.state).success
    result = engine.stages["generate_code"].execute(engine.state)

    assert result.success
    generated_root = workspace / ".attest" / "generated_repo"
    generated_cpp = generated_root / "math" / "add" / "tests" / "ut" / "op_host" / "test_add_tiling_attest.cpp"
    baseline_cpp = repo_root / "math" / "add" / "tests" / "ut" / "op_host" / "test_add_tiling.cpp"

    assert generated_cpp.exists()
    assert "CASE_01" in generated_cpp.read_text(encoding="utf-8")
    assert "CASE_01" not in baseline_cpp.read_text(encoding="utf-8")


def test_codegen_resumes_placeholder_header_footer(tmp_path):
    repo_root = _make_repo(tmp_path)
    workspace = tmp_path / "workspace"
    engine = _make_engine(repo_root, workspace)

    assert engine.stages["understand_function"].execute(engine.state).success
    assert engine.stages["generate_requirements"].execute(engine.state).success
    assert engine.stages["design_test_plan"].execute(engine.state).success

    stage = engine.stages["generate_code"]
    plan = json.loads(engine.state.load_artifact("test_plan.json"))
    project_root = repo_root
    cmake_entry = next(item for item in plan["files"] if item["kind"] == "cmake")
    stage._ensure_skeleton(project_root, cmake_entry, [])

    targets = stage._select_target_blocks(project_root, engine.state, plan, cmake_entry, {}, created=False)

    assert "HEADER" in targets
    assert "FOOTER" in targets


def test_execution_and_analysis_compare_coverage(tmp_path):
    repo_root = _make_repo(tmp_path, with_op_api=True, with_existing_ut=True)
    workspace = tmp_path / "workspace"
    engine = _make_engine(repo_root, workspace)

    assert engine.stages["understand_function"].execute(engine.state).success
    assert engine.stages["generate_requirements"].execute(engine.state).success
    assert engine.stages["design_test_plan"].execute(engine.state).success
    assert engine.stages["generate_code"].execute(engine.state).success

    exec_result = engine.stages["execute_tests"].execute(engine.state)
    analysis_result = engine.stages["analyze_results"].execute(engine.state)

    assert exec_result.success
    assert analysis_result.success

    coverage = json.loads(engine.state.load_artifact("coverage_summary.json"))
    assert coverage["mode"] == "before_after_compare"
    assert coverage["per_layer"]["op_host"]["baseline"]["line_coverage"] == 70.0
    assert coverage["per_layer"]["op_host"]["generated"]["line_coverage"] == 86.5
    assert coverage["per_layer"]["op_api"]["baseline"]["line_coverage"] == 61.0
    assert coverage["per_layer"]["op_api"]["generated"]["line_coverage"] == 84.0
    assert coverage["overall"]["generated_ge_baseline"] is True

    analysis_plan = json.loads(engine.state.load_artifact("analysis_plan.json"))
    assert analysis_plan["status"] == "success"
    assert "not lower than the baseline" in engine.state.auto_stop_reason


def test_ascend_end_to_end_smoke_compare_mode(tmp_path):
    repo_root = _make_repo(tmp_path, with_op_api=True, with_existing_ut=True)
    workspace = tmp_path / "workspace"
    engine = _make_engine(repo_root, workspace, epochs=1)

    engine.run(mode="full-auto")

    report_path = workspace / ".attest" / "artifacts" / "generate_report" / "current_final_report.md"
    assert report_path.exists()

    generated_root = workspace / ".attest" / "generated_repo"
    generated_cpp = generated_root / "math" / "add" / "tests" / "ut" / "op_host" / "test_add_tiling_attest.cpp"
    baseline_cpp = repo_root / "math" / "add" / "tests" / "ut" / "op_host" / "test_add_tiling.cpp"

    entries = build_block_entries(generated_cpp)
    assert any(entry["block_id"] == "CASE_01" and entry["status"] == "bounded" for entry in entries)
    assert "CASE_01" not in baseline_cpp.read_text(encoding="utf-8")
