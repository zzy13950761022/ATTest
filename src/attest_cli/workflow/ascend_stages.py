"""
Ascend-specific workflow stages.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import dataclass
import json
import re
import shlex
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..ascend import AscendSkillProvider
from ..block_utils import (
    build_block_entries,
    build_block_index_json,
    detect_comment_style,
    end_marker,
    placeholder_marker,
    start_marker,
)
from ..config import load_config
from ..session import append_message as _raw_append_message
from ..tools import ToolContext
from ..utils import ensure_parent
from .stage import Stage, StageConfig, StageResult


# Guards the shared session-history JSONL writes (append_message) when multiple
# per-layer generator agents run concurrently in the parallel generation path.
_APPEND_LOCK = threading.Lock()


def append_message(*args, **kwargs):
    """Thread-safe wrapper around session.append_message.

    The parallel generation path runs multiple _run_llm_session loops in worker
    threads that all append to the same session JSONL; serialize those writes.
    """
    with _APPEND_LOCK:
        return _raw_append_message(*args, **kwargs)



LAYER_ORDER = {
    "op_host": 0,
    "op_api": 1,
    "op_kernel": 2,
    "op_kernel_aicpu": 3,
}

LAYER_COMMAND_KEYS = {
    "op_host": "compile_ophost",
    "op_api": "compile_opapi",
    "op_kernel": "compile_opkernel",
    "op_kernel_aicpu": "compile_opkernel_aicpu",
}

LAYER_FLAG_NAMES = {
    "op_host": "ophost",
    "op_api": "opapi",
    "op_kernel": "opkernel",
    "op_kernel_aicpu": "opkernel_aicpu",
}


def _json_load(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not value:
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _load_json_artifact(state, name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _json_load(state.load_artifact(name), default or {})


def _render_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


_GTEST_RESULT_KWS = (
    "[  PASSED  ]", "[  FAILED  ]", "YOU HAVE", "FAILED TESTS",
    "tests from", "tests ran",
)
_LCOV_SUMMARY_KWS = (
    "Summary coverage rate", "lines......", "functions..", "branches...",
    "uncovered_lines", "operator lcov summary",
)


_NOISE_KWS = (
    "Processing file", "-- The C ", "-- Detecting", "-- Check for working",
    "-- Detect compile", "-- Detect CXX", "-- Find", "-- Configuring done",
    "-- Generating done", "HEAD is now at", "Scanning dependencies",
    "Linking CXX", "CMAKE_ARGS:", "[  0%]", "[  1%]", "[  2%]", "[  3%]",
    "Building CXX object", "Built target", "Consolidate compiler",
)


def _extract_execution_summary(log_text: str, max_chars: int = 8000) -> str:
    """Extract useful test-result / coverage sections from an execution log.

    Keeps: gtest pass/fail lines, lcov coverage summary, uncovered lines.
    Skips build-system noise (cmake progress, genhtml file processing).
    Falls back to the last 200 lines of the log if no markers match.
    """
    lines = log_text.splitlines()
    kept: List[str] = []
    capture_cov = False
    for line in lines:
        if any(kw in line for kw in _NOISE_KWS):
            capture_cov = False
            continue
        is_gtest = any(kw in line for kw in _GTEST_RESULT_KWS)
        is_cov = any(kw in line for kw in _LCOV_SUMMARY_KWS)
        if is_gtest:
            kept.append(line)
            capture_cov = False
        elif is_cov:
            kept.append(line)
            capture_cov = "operator lcov summary" in line or "uncovered_lines" not in line
        elif capture_cov:
            kept.append(line)
            if "uncovered_lines" in line or "===" in line:
                capture_cov = False
    if not kept:
        kept = lines[-200:] if len(lines) > 200 else lines
    result = "\n".join(kept)
    if len(result) > max_chars:
        result = result[-max_chars:]
    return result


def _compress_old_messages(messages: List[Dict[str, Any]], keep_recent: int = 60) -> List[Dict[str, Any]]:
    """Sliding-window compression: keep the system/initial user prompt + last `keep_recent` messages."""
    if len(messages) <= keep_recent + 2:
        return messages
    # Always keep messages[0] (initial user prompt) and the most recent `keep_recent` messages
    head = messages[:1]
    tail = messages[-keep_recent:]
    dropped = len(messages) - 1 - keep_recent
    summary_msg = {
        "role": "user",
        "content": (
            f"[Context compression: {dropped} earlier messages omitted to stay within context limits. "
            "Your prior file edits are already on disk — use `ls`/`cat` to inspect current state.]"
        ),
    }
    return head + [summary_msg] + tail


def _build_ws_sig_section(layer_id: str, slim_context: Dict[str, Any]) -> str:
    if layer_id != "op_api":
        return ""
    ws_sigs = slim_context.get("workspace_signatures") or []
    if not ws_sigs:
        return ""
    lines = [
        "OP_API_UT Function Signatures (CRITICAL for correct INPUT/OUTPUT macro usage):",
        "Use these signatures to determine how many parameters go into INPUT() and OUTPUT().",
        "The `OP_API_UT(api, INPUT(...), OUTPUT(...))` macro calls `api##GetWorkspaceSize` with:",
        "  - INPUT params → first N positional arguments",
        "  - OUTPUT params → next M positional arguments",
        "  - framework auto-appends `&workspaceSize` and `&executor`",
        "```json",
        json.dumps(ws_sigs, indent=2, ensure_ascii=False),
        "```",
        "**Example**: if the signature shows `input_params: [self, other]` and `output_params: [out]`,",
        "use `OP_API_UT(aclnnXxx, INPUT(self_desc, other_desc), OUTPUT(out_desc))`.",
        "**For Inplace variants**: if `output_params` is `[]`, use `OUTPUT()` (empty).",
        "**For Inplace variants with selfRef**: selfRef is an input, NOT an output — pass it in INPUT only.",
    ]
    return "\n".join(lines)


_FEW_SHOT_OP_HOST_INFERSHAPE = '''## FEW-SHOT EXAMPLE: op_host InferShape test (CORRECT pattern)

The CORRECT InferShape test pattern uses `InfershapeContextPara` + `ExecuteTestCase`.
You MUST use this exact pattern — do NOT call `CreateInferShapeContext`, `gert::InferShape`,
`gert::InferShapeContextFaker().Build()`, or any other API that is not in this example.

```cpp
// ==== BLOCK:HEADER START ====
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class MyOpInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "MyOpInferShape SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "MyOpInferShape TearDown" << std::endl; }
};
// ==== BLOCK:HEADER END ====
// ==== BLOCK:CASE_01 START ====
TEST_F(MyOpInferShape, case_01_valid_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "MyOpName",
    {
      {{{m, n}, {m, n}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1
      {{{n, k}, {n, k}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 2
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{m, k}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_01 END ====
```

Key rules:
- Include `"infershape_context_faker.h"` and `"infershape_case_executor.h"` (NOT `base/registry/...`)
- Use `gert::InfershapeContextPara` constructor (NOT `gert::CreateInferShapeContext`)
- Use free function `ExecuteTestCase(para, ge::GRAPH_SUCCESS/FAILED, expectedShapes)`
- For op_host tiling: use `gert::TilingContextPara` analogously (ask the skill docs if unsure)
'''

_FEW_SHOT_OP_API = '''## FEW-SHOT EXAMPLE: op_api test (CORRECT pattern)

The CORRECT op_api test pattern uses `OP_API_UT`, `TensorDesc`, `ScalarDesc`, and `TestGetWorkspaceSize`.
You MUST use these macros — do NOT guess API names like `CreateInferShapeContext` or similar.

```cpp
// ==== BLOCK:HEADER START ====
#include <gtest/gtest.h>
#include "op_api_ut_helper.h"

class myop_test : public testing::Test {
 protected:
  static void SetUpTestCase() {}
  static void TearDownTestCase() {}
};
// ==== BLOCK:HEADER END ====
// ==== BLOCK:CASE_01 START ====
TEST_F(myop_test, case_01_nullptr_input) {
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMyOp, INPUT((aclTensor*)nullptr), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}
// ==== BLOCK:CASE_01 END ====
// ==== BLOCK:CASE_02 START ====
TEST_F(myop_test, case_02_valid_float) {
    auto self_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMyOp, INPUT(self_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_02 END ====
```

Common dtype values: `ACL_FLOAT`, `ACL_FLOAT16`, `ACL_INT32`, `ACL_INT64`, `ACL_INT8`, `ACL_UINT8`, `ACL_BOOL`.
BITWISE ops ONLY accept integer types (`ACL_INT8`, `ACL_INT16`, `ACL_INT32`, `ACL_INT64`, `ACL_UINT8`, `ACL_BOOL`) — do NOT pass `ACL_FLOAT` to bitwise ops.
Common formats: `ACL_FORMAT_ND`, `ACL_FORMAT_NCHW`, `ACL_FORMAT_NHWC`.
'''


def _get_few_shot_examples(layer_id: str) -> str:
    """Return a few-shot example block for the given layer, or empty string."""
    if layer_id == "op_host":
        return _FEW_SHOT_OP_HOST_INFERSHAPE
    if layer_id == "op_api":
        return _FEW_SHOT_OP_API
    return ""


def _patch_build_sh_for_isolation(target_root: Path) -> None:
    build_sh = target_root / "build.sh"
    if not build_sh.exists():
        return
    content = build_sh.read_text(encoding="utf-8")
    if '${BUILD_PATH:-' in content:
        return
    content = re.sub(
        r'export BUILD_PATH="\$\{BASE_PATH\}/build"',
        'export BUILD_PATH=${BUILD_PATH:-"${BASE_PATH}/build"}',
        content,
    )
    content = re.sub(
        r'export BUILD_OUT_PATH="\$\{BASE_PATH\}/build_out"',
        'export BUILD_OUT_PATH=${BUILD_OUT_PATH:-"${BASE_PATH}/build_out"}',
        content,
    )
    build_sh.write_text(content, encoding="utf-8")


_CANN_CMAKE_PREP_MARKER = "function/prepare.cmake"


def _ensure_cann_cmake_available(target_root: Path) -> bool:
    target = target_root / "third_party" / "cann-cmake"
    if (target / _CANN_CMAKE_PREP_MARKER).exists():
        return True

    build_deps = target_root / "build" / "_deps" / "cann-cmake-src"
    candidates: List[Path] = []
    if (build_deps / _CANN_CMAKE_PREP_MARKER).exists():
        candidates.append(build_deps)

    for sibling in (target_root.parent, target_root):
        if sibling and sibling.is_dir():
            for child in sibling.iterdir():
                if not child.is_dir():
                    continue
                for candidate_rel in (
                    Path("build") / "_deps" / "cann-cmake-src",
                    Path(".attest") / "generated_repo" / "build" / "_deps" / "cann-cmake-src",
                ):
                    candidate = child / candidate_rel
                    if (candidate / _CANN_CMAKE_PREP_MARKER).exists():
                        candidates.append(candidate)

    for known in (
        Path("/mnt/fangcr/workspace-baseline/B3_opencode_skill/build/_deps/cann-cmake-src"),
        Path("/mnt/fangcr/workspace-baseline/B2_opencode_plain/build/_deps/cann-cmake-src"),
    ):
        if (known / _CANN_CMAKE_PREP_MARKER).exists():
            candidates.append(known)

    for backup_base in (
        Path("/mnt/fangcr/ops-math-round1-backup-20260515_231645"),
        Path("/mnt/fangcr/ops-math-round2-backup-20260517_221941"),
    ):
        backup_src = backup_base / "build" / "_deps" / "cann-cmake-src"
        if (backup_src / _CANN_CMAKE_PREP_MARKER).exists():
            candidates.append(backup_src)

    for src in candidates:
        try:
            shutil.copytree(src, target, dirs_exist_ok=True)
            if (target / _CANN_CMAKE_PREP_MARKER).exists():
                return True
        except Exception:
            continue
    return False


def _ordered_files(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    files = plan.get("files") or []
    return sorted(
        [item for item in files if isinstance(item, dict)],
        key=lambda item: (
            0 if item.get("kind") == "cmake" else 1,
            LAYER_ORDER.get(str(item.get("layer_id")), 99),
            str(item.get("path", "")),
        ),
    )


def _cases_by_file(plan: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in plan.get("cases") or []:
        if isinstance(case, dict):
            buckets[str(case.get("file_id", ""))].append(case)
    for values in buckets.values():
        values.sort(key=lambda item: str(item.get("block_id", "")))
    return buckets


def _smoke_set(plan: Dict[str, Any]) -> List[str]:
    return [item for item in (plan.get("smoke_set") or []) if isinstance(item, str)]


def _deferred_set(plan: Dict[str, Any]) -> List[str]:
    return [item for item in (plan.get("deferred_set") or []) if isinstance(item, str)]


def _path_to_file_id(plan: Dict[str, Any]) -> Dict[str, str]:
    mapping = {}
    for entry in plan.get("files") or []:
        if isinstance(entry, dict):
            mapping[str(entry.get("path", ""))] = str(entry.get("file_id", ""))
    return mapping


def _extract_existing_test_scenarios(ut_file_path: Path, max_scenarios: int = 30) -> List[str]:
    if not ut_file_path.exists():
        return []
    try:
        content = ut_file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    scenarios = []
    test_name_pattern = re.compile(r'TEST[_F]?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)')
    for match in test_name_pattern.finditer(content):
        suite = match.group(1)
        case = match.group(2)
        start = match.end()
        end = min(start + 500, len(content))
        body = content[start:end]
        dtypes = set()
        for dtype in ["DT_FLOAT", "DT_FLOAT16", "DT_BF16", "DT_INT32", "DT_INT64", "DT_INT8",
                       "DT_INT16", "DT_UINT8", "DT_BOOL", "DT_COMPLEX64", "DT_COMPLEX128",
                       "DT_DOUBLE", "DT_UINT16", "DT_UINT32", "DT_UINT64"]:
            if dtype in body:
                dtypes.add(dtype.replace("DT_", ""))
        formats = set()
        for fmt in ["FORMAT_ND", "FORMAT_NCHW", "FORMAT_NHWC", "FORMAT_NC1HWC0",
                     "FORMAT_NCDHW", "FORMAT_NDHWC", "FORMAT_HWCN"]:
            if fmt in body:
                formats.add(fmt.replace("FORMAT_", ""))
        tags = []
        if "nullptr" in body or "NULL" in body:
            tags.append("nullptr")
        if "empty" in body.lower() or "shape_0" in body or "{0" in body:
            tags.append("empty")
        if "shape_mismatch" in body.lower() or "mismatch" in body.lower():
            tags.append("shape_mismatch")
        if "invalid" in body.lower() or "error" in body.lower():
            tags.append("invalid")
        if "broadcast" in body.lower():
            tags.append("broadcast")
        if "inplace" in body.lower() or "Inplace" in body:
            tags.append("inplace")
        desc_parts = []
        if dtypes:
            desc_parts.append("dtypes=" + ",".join(sorted(dtypes)[:4]))
        if formats:
            desc_parts.append("fmts=" + ",".join(sorted(formats)[:3]))
        if tags:
            desc_parts.append(",".join(tags[:3]))
        scenario = f"{suite}.{case}"
        if desc_parts:
            scenario += f" ({'; '.join(desc_parts)})"
        scenarios.append(scenario)
        if len(scenarios) >= max_scenarios:
            break
    return scenarios


def _default_analysis_plan() -> Dict[str, Any]:
    return {
        "status": "not_fully_passed",
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "collection_errors": False,
        "block_limit": 12,
        "failures": [],
        "deferred": [],
        "stop_recommended": False,
        "stop_reason": "",
    }


def _generation_mode(meta: Dict[str, Any]) -> str:
    return str(meta.get("generation_mode", "ut_generate"))


def _coverage_mode(meta: Dict[str, Any]) -> str:
    return str(meta.get("coverage_mode") or meta.get("mode") or "single_run")


def _is_enhance_mode(meta: Dict[str, Any]) -> bool:
    return _generation_mode(meta) == "ut_enhance"


def _is_compare_mode(meta: Dict[str, Any]) -> bool:
    return _coverage_mode(meta) == "before_after_compare"


def _layer_coverage_threshold(meta: Optional[Dict[str, Any]] = None) -> float:
    if isinstance(meta, dict) and _is_compare_mode(meta):
        return 85.0
    return 80.0


def _overall_coverage_threshold(meta: Optional[Dict[str, Any]] = None) -> float:
    if isinstance(meta, dict) and _is_compare_mode(meta):
        return 90.0
    return 80.0


def _coverage_stop_policy(meta: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_config().get("profiles", {}).get("ascend_ut", {}).get("coverage_stop_policy", {})
    enhance_compare = _is_compare_mode(meta) and _is_enhance_mode(meta)
    default_stop_on_threshold = not enhance_compare
    stop_key = "enhance_stop_on_threshold" if enhance_compare else "generate_stop_on_threshold"
    if stop_key in cfg:
        stop_on_threshold = bool(cfg.get(stop_key))
    elif "stop_on_threshold" in cfg:
        stop_on_threshold = bool(cfg.get("stop_on_threshold"))
    else:
        stop_on_threshold = default_stop_on_threshold
    try:
        patience = int(cfg.get("patience_no_improvement", 1))
    except (TypeError, ValueError):
        patience = 1
    return {
        "stop_on_threshold": stop_on_threshold,
        "patience_no_improvement": max(1, patience),
    }


def _operator_source_patterns(meta: Dict[str, Any], layer: str) -> List[str]:
    category = str(meta.get("category") or "")
    op_name = str(meta.get("op_name") or "")
    if not category or not op_name:
        return []

    root = f"*/{category}/{op_name}"
    if layer == "op_host":
        return [
            f"{root}/op_host/*",
        ]
    if layer == "op_api":
        return [f"{root}/op_api/*", f"{root}/op_host/op_api/*"]
    return []


def _operator_remove_patterns(meta: Dict[str, Any], layer: str, infra_overrides: Optional[Dict[str, Any]] = None) -> List[str]:
    category = str(meta.get("category") or "")
    op_name = str(meta.get("op_name") or "")
    if not category or not op_name:
        return []

    root = f"*/{category}/{op_name}"
    if layer == "op_host":
        if infra_overrides and infra_overrides.get("op_host", {}).get("skip_remove"):
            return []
        return [f"{root}/op_host/op_api/*"]
    return []


def _detect_op_host_layout(op_dir: Path) -> Dict[str, Any]:
    op_host = op_dir / "op_host"
    if not op_host.is_dir():
        return {"exists": False}
    cpp_files = list(op_host.glob("*.cpp"))
    has_op_api_subdir = (op_host / "op_api").is_dir()
    op_api_has_cpp = any((op_host / "op_api").glob("*.cpp")) if has_op_api_subdir else False
    has_real_host_code = bool(cpp_files)
    is_pure_opapi = has_op_api_subdir and not has_real_host_code
    return {
        "exists": True,
        "op_host_cpp_files": [f.name for f in cpp_files],
        "has_op_api_subdir": has_op_api_subdir,
        "op_api_has_cpp": op_api_has_cpp,
        "has_real_host_code": has_real_host_code,
        "is_pure_opapi": is_pure_opapi,
    }


def _probe_cmake_info(op_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"cmake_style": "unknown", "targets": {}, "macros": {}, "warnings": []}
    top_cmake = op_dir / "CMakeLists.txt"
    if top_cmake.exists():
        text = top_cmake.read_text(encoding="utf-8", errors="replace")
        if "add_all_modules_sources" in text:
            result["cmake_style"] = "new_add_all_modules_sources"
        elif "add_modules_sources" in text:
            result["cmake_style"] = "old_add_modules_sources"
        elif "foreach" in text and "add_subdirectory" in text:
            result["cmake_style"] = "foreach_add_subdirectory"
    op_host_cmake = op_dir / "op_host" / "CMakeLists.txt"
    if op_host_cmake.exists():
        text = op_host_cmake.read_text(encoding="utf-8", errors="replace")
        if "add_modules_sources" in text:
            m = re.search(r"add_modules_sources\(([^)]+)\)", text)
            if m:
                result["macros"]["op_host_build"] = f"add_modules_sources({m.group(1).strip()})"
        if "add_all_modules_sources" in text:
            result["macros"]["op_host_build"] = "add_all_modules_sources"
    ut_op_host_cmake = op_dir / "tests" / "ut" / "op_host" / "CMakeLists.txt"
    ut_op_host_exists = ut_op_host_cmake.exists()
    result["test_cmake"] = {"op_host_exists": ut_op_host_exists}
    if ut_op_host_exists:
        text = ut_op_host_cmake.read_text(encoding="utf-8", errors="replace")
        if "OP_TILING_MODULE_NAME" in text or "OP_INFERSHAPE_MODULE_NAME" in text:
            result["targets"]["op_host_ut"] = ["${OP_TILING_MODULE_NAME}", "${OP_INFERSHAPE_MODULE_NAME}"]
            result["test_register_macro"] = "add_modules_ut_sources"
        for m in re.finditer(r"\$\{(\w+)\}_cases_obj", text):
            result["targets"].setdefault("op_host_cases", []).append(f"${{{m.group(1)}}}_cases_obj")
    ut_op_api_cmake = op_dir / "tests" / "ut" / "op_api" / "CMakeLists.txt"
    ut_op_api_exists = ut_op_api_cmake.exists()
    result["test_cmake"]["op_api_exists"] = ut_op_api_exists
    if ut_op_api_exists:
        text = ut_op_api_cmake.read_text(encoding="utf-8", errors="replace")
        if "OP_API_MODULE_NAME" in text:
            result["targets"]["op_api_ut"] = ["${OP_API_MODULE_NAME}"]
    ut_op_host_op_api_cmake = op_dir / "tests" / "ut" / "op_host" / "op_api" / "CMakeLists.txt"
    if ut_op_host_op_api_cmake.exists():
        result["test_cmake"]["op_host_op_api_exists"] = True
        result["warnings"].append(
            "tests/ut/op_host/op_api/CMakeLists.txt exists — UT follows nested op_host/op_api source layout"
        )
    return result


def _collect_infrastructure_context(
    state, op_dir: Path, context: Dict[str, Any]
) -> Dict[str, Any]:
    op_name = state.op_name
    category = state.category
    layout = _detect_op_host_layout(op_dir)
    cmake_info = _probe_cmake_info(op_dir)
    enabled_layers = list(context.get("enabled_layers", []))
    lcov_overrides: Dict[str, Dict[str, Any]] = {}
    if layout.get("is_pure_opapi") and "op_host" in enabled_layers:
        lcov_overrides["op_host"] = {
            "skip_remove": True,
            "reason": "op_host/ contains only op_api/ subdir — lcov --remove would wipe all records",
        }
    trackable: Dict[str, List[str]] = {}
    op_host = op_dir / "op_host"
    if op_host.is_dir() and layout.get("has_real_host_code"):
        trackable["op_host"] = [f.name for f in op_host.glob("*.cpp")]
    op_api = op_dir / "op_api"
    if op_api.is_dir():
        trackable["op_api"] = [f.name for f in op_api.glob("*.cpp")]
    elif (op_host / "op_api").is_dir():
        trackable["op_api"] = [f.name for f in (op_host / "op_api").glob("*.cpp")]
    warnings: List[str] = list(cmake_info.get("warnings", []))
    if layout.get("is_pure_opapi"):
        warnings.append(
            f"op_host/ for '{op_name}' contains no infershape/tiling .cpp files — "
            "only op_api/ nesting. op_host coverage may be meaningless."
        )
    if not cmake_info["test_cmake"].get("op_host_exists"):
        warnings.append("tests/ut/op_host/CMakeLists.txt does NOT exist — will be generated at runtime")
    if not cmake_info["test_cmake"].get("op_api_exists"):
        warnings.append("tests/ut/op_api/CMakeLists.txt does NOT exist — will be generated at runtime")

    build_commands = context.get("build_commands", {})
    build_cmd_template = ""
    if build_commands:
        cov = build_commands.get("combined_coverage", "")
        compile = build_commands.get("combined_compile", "")
        build_cmd_template = cov or compile or ""

    return {
        "op_name": op_name,
        "category": category,
        "op_host_layout": layout,
        "cmake_info": cmake_info,
        "trackable_sources": trackable,
        "lcov_overrides": lcov_overrides,
        "build_cmd_template": build_cmd_template[:500],
        "warnings": warnings,
    }


def _normalize_case_block_id(raw_value: str) -> str:
    match = re.search(r"(CASE_\d+)", raw_value)
    if match:
        return match.group(1)
    return raw_value


def _coverage_target_actually_failed(log_text: str) -> bool:
    lowered = log_text.lower()
    if "built target generate_ops_cpp_cov" in lowered:
        return False
    cov_build_fail = re.search(
        r"gmake\[\d+\]:\s*\*\*\*.*generate_ops_cpp_cov.*error",
        lowered,
    )
    if cov_build_fail:
        return True
    error_255_with_cov = re.search(
        r"generate_ops_cpp_cov.*error\s*255|error\s*255.*generate_ops_cpp_cov",
        lowered,
    )
    if error_255_with_cov:
        return True
    genhtml_error = "genhtml: error" in lowered and "no valid records found in tracefile" in lowered
    if genhtml_error:
        return True
    lcov_no_data = re.search(r"lines\.+:\s*no data found", lowered) is not None
    if lcov_no_data and "built target generate_ops_cpp_cov" not in lowered:
        return True
    return False


def _infer_execution_error_type(log_text: str) -> str:
    lowered = log_text.lower()
    has_ok_tests = "[  ok  ]" in lowered or "[       ok ]" in lowered
    has_test_failure = (
        "[  failed  ]" in lowered
        or "segmentation fault" in lowered
        or "core dumped" in lowered
    )
    if has_test_failure and not ("[  failed  ]" in lowered):
        if "segmentation fault" in lowered or "core dumped" in lowered:
            if has_ok_tests and (
                "--cov" in lowered
                or "gcov" in lowered
                or "coverage" in lowered
            ):
                return "CoverageInstrumentationError"
    has_compile_failure = (
        "undefined reference" in lowered
        or "collect2:" in lowered
        or "cmake error" in lowered
        or re.search(r"\.(?:c|cc|cpp|cxx|h|hpp):[0-9]+:[0-9]+:\s+error:", lowered) is not None
        or re.search(r"\bfatal error:", lowered) is not None
    )
    has_cov_target_failure = _coverage_target_actually_failed(log_text)
    lcov_specific_failure = (
        "lcov_extract_no_match" in lowered
        or "lcov_source_missing" in lowered
        or "lcov: error" in lowered
        or "libgcov profiling error" in lowered
    )
    has_coverage_failure = has_cov_target_failure or lcov_specific_failure
    if has_coverage_failure:
        if not has_compile_failure and not has_test_failure:
            return "CoverageCollectionError"
    if "assertionerror" in lowered:
        return "AssertionError"
    if has_test_failure:
        return "TestFailure"
    if has_compile_failure:
        return "CompilationError"
    if "error:" in lowered:
        if "gmake[" in lowered and "math_op_host_ut" in lowered and "run ops op_host utest" in lowered:
            return "TestFailure"
        return "CompilationError"
    if "gmake[" in lowered and ("math_op_host_ut" in lowered or "math_op_api_ut" in lowered):
        return "TestFailure"
    return "CompilationError"


def _coverage_error_reason(log_text: str) -> Optional[str]:
    lowered = log_text.lower()
    if _coverage_target_actually_failed(log_text):
        if re.search(r"lines\.+:\s*no data found", lowered):
            return "lcov summary contains no coverage data (no instrumented code executed)"
        if "genhtml: error" in lowered and "no valid records" in lowered:
            return "coverage tracefile is empty (no tests exercised instrumented code)"
        return "coverage target generate_ops_cpp_cov failed"
    if "lcov_extract_no_match" in lowered:
        return "operator lcov extract matched no source files"
    if "lcov_source_missing" in lowered:
        return "lcov source info file is missing"
    if "lcov: error" in lowered:
        if "no valid records found" in lowered:
            return "ZeroTestsRegistered"
        return "lcov/genhtml reported an error"
    if re.search(r"lines\.+:\s*no data found", lowered):
        return "lcov summary contains no coverage data"
    if "libgcov profiling error" in lowered:
        return "gcov profile data is stale or mixed across builds"
    return None


def _generated_repo_root(state) -> Path:
    return state.workspace / ".attest" / "generated_repo"


def _generated_operator_root(state, context: Dict[str, Any]) -> Path:
    return _generated_repo_root(state) / str(context.get("relative_op_dir", state.category + "/" + state.op_name))


def _llm_project_root(state, context: Dict[str, Any]) -> Path:
    if _is_enhance_mode(context):
        return _generated_repo_root(state)
    return Path(state.project_root)


def _read_file_truncated(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError):
        return ""
    total = len(text)
    text = re.sub(r"^\s*(?:/\*.*?\*/|//.*?(?:\n|$))+", "", text, count=1, flags=re.DOTALL)
    text = text.lstrip()
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated, total {total} chars]"
    return text


def _read_op_source_code(operator_dir: Path, op_name: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not operator_dir.exists():
        return result
    name_lower = op_name.lower()
    op_host = operator_dir / "op_host"
    op_api = operator_dir / "op_api"
    for candidate in [
        op_host / f"{name_lower}_def.cpp",
    ]:
        if candidate.exists():
            result["op_host_def"] = _read_file_truncated(candidate, 800)
            break
    for candidate in [
        op_host / f"{name_lower}_infershape.cpp",
    ]:
        if candidate.exists():
            result["op_host_infershape"] = _read_file_truncated(candidate, 700)
            break
    for candidate in [
        op_api / f"{name_lower}.h",
        op_api / f"aclnn_{name_lower}.h",
    ]:
        if candidate.exists():
            result["op_api_header"] = _read_file_truncated(candidate, 900)
            break
    return {k: v for k, v in result.items() if v}


def _parse_exit_payload(raw_value: Any) -> Dict[str, int]:
    if isinstance(raw_value, dict):
        return {str(key): int(value) for key, value in raw_value.items()}

    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return {"overall": 1}

    try:
        parsed = json.loads(raw_text)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        payload: Dict[str, int] = {}
        for key, value in parsed.items():
            try:
                payload[str(key)] = int(value)
            except Exception:
                continue
        return payload or {"overall": 1}

    try:
        return {"overall": int(raw_text)}
    except Exception:
        return {"overall": 1}


def _rewrite_context_for_project_root(context: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    rewritten = dict(context)
    rewritten["project_root"] = str(project_root)
    relative_op_dir = str(rewritten.get("relative_op_dir", ""))
    original_operator_dir = str(rewritten.get("operator_dir", ""))
    rewritten_operator_dir = str(project_root / relative_op_dir) if relative_op_dir else original_operator_dir
    if rewritten_operator_dir:
        rewritten["operator_dir"] = rewritten_operator_dir
        rewritten["tests_root"] = str(Path(rewritten_operator_dir) / "tests" / "ut")

    layers = rewritten.get("layers")
    if isinstance(layers, dict) and rewritten_operator_dir and original_operator_dir:
        rebuilt_layers: Dict[str, Any] = {}
        original_root = Path(original_operator_dir)
        rewritten_root = Path(rewritten_operator_dir)
        for layer_name, meta in layers.items():
            if not isinstance(meta, dict):
                rebuilt_layers[layer_name] = meta
                continue
            rebuilt_meta = dict(meta)
            raw_path = str(meta.get("path", ""))
            try:
                rel = Path(raw_path).relative_to(original_root)
                rebuilt_meta["path"] = str(rewritten_root / rel)
            except Exception:
                rebuilt_meta["path"] = str(rewritten_root / layer_name)
            rebuilt_layers[layer_name] = rebuilt_meta
        rewritten["layers"] = rebuilt_layers

    reference_context = rewritten.get("reference_context")
    if isinstance(reference_context, dict):
        rebuilt_reference = dict(reference_context)
        rebuilt_current_operator_ut: Dict[str, List[str]] = {}
        for layer_name, values in (reference_context.get("current_operator_ut") or {}).items():
            if isinstance(values, list):
                rebuilt_current_operator_ut[str(layer_name)] = [str(item) for item in values if isinstance(item, str)]
        rebuilt_reference["current_operator_ut"] = rebuilt_current_operator_ut
        rewritten["reference_context"] = rebuilt_reference

    return rewritten


def _sanitize_cmake_footer(content: str) -> tuple:
    """Strip LLM-generated duplicate registration lines from CMake FOOTER."""
    footer_start = content.find("# ==== BLOCK:FOOTER START ====")
    footer_end = content.find("# ==== BLOCK:FOOTER END ====")
    if footer_start == -1 or footer_end == -1 or footer_end <= footer_start:
        return content, False
    before = content[: footer_start]
    after = content[footer_end:]
    header_marker = "# ==== BLOCK:FOOTER START ====\n"
    footer_body_start = content.find("\n", footer_start) + 1
    footer_body = content[footer_body_start:footer_end]
    bad_keywords = ["add_modules_ut_sources", "add_library",
                    "set(OP_API_TEST_SOURCES", "set(OP_HOST_TEST_SOURCES"]
    kept = [l for l in footer_body.splitlines() if not any(k in l for k in bad_keywords)]
    new_body = "\n".join(kept)
    if len(kept) < len(footer_body.splitlines()):
        return before + header_marker + new_body + ("\n" if new_body and not new_body.endswith("\n") else "") + after, True
    return content, False


def _sanitize_context_for_generation(context: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(context)
    generated_project_root = sanitized.get("generated_project_root")
    if generated_project_root:
        sanitized = _rewrite_context_for_project_root(sanitized, Path(str(generated_project_root)))

    sanitized["reference_ut_paths"] = []
    sanitized["existing_ut_files"] = []
    sanitized["existing_ut_by_layer"] = {}
    if isinstance(sanitized.get("reference_context"), dict):
        sanitized["reference_context"] = dict(sanitized["reference_context"])
        sanitized["reference_context"]["current_operator_ut"] = {}
    return sanitized


def _reference_context(meta: Dict[str, Any]) -> Dict[str, Any]:
    value = meta.get("reference_context")
    return value if isinstance(value, dict) else {}


def _paths_for_layer(paths: List[Any], layer_id: str) -> List[str]:
    token = f"/{layer_id}/"
    return [str(item) for item in paths if isinstance(item, str) and token in item]


def _shared_harness_for_layer(meta: Dict[str, Any], layer_id: str) -> List[str]:
    shared = _reference_context(meta).get("shared_test_harness") or []
    values = [path for path in _paths_for_layer(shared, layer_id)]
    values.extend([str(item) for item in shared if isinstance(item, str) and "/common/" in item])
    return list(dict.fromkeys(values))[:8]


def _shared_helpers_for_layer(meta: Dict[str, Any], layer_id: str) -> List[str]:
    if layer_id != "op_host":
        return []
    helpers = _reference_context(meta).get("shared_helpers") or []
    return [str(item) for item in helpers if isinstance(item, str)][:8]


def _operator_impl_for_layer(meta: Dict[str, Any], layer_id: str) -> List[str]:
    impl_files = _reference_context(meta).get("operator_impl_files") or []
    return _paths_for_layer(impl_files, layer_id)[:10]


def _similar_examples_for_layer(meta: Dict[str, Any], layer_id: str) -> List[str]:
    examples = _reference_context(meta).get("similar_examples") or []
    values = _paths_for_layer(examples, layer_id)
    return values[:6]


def _current_operator_ut_for_layer(meta: Dict[str, Any], layer_id: str) -> List[str]:
    current_ut = _reference_context(meta).get("current_operator_ut") or {}
    values = current_ut.get(layer_id) if isinstance(current_ut, dict) else []
    return [str(item) for item in values if isinstance(item, str)][:8]


class AscendBaseStage(Stage):
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)

    def _skill_provider(self, state) -> AscendSkillProvider:
        skill_root = getattr(state, "skill_root", "") or load_config().get("profiles", {}).get("ascend_ut", {}).get("skill_root", "")
        return AscendSkillProvider(skill_root=skill_root)

    def get_config(self) -> StageConfig:
        return self.config


class AscendInspectOperatorStage(AscendBaseStage):
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="understand_function",
            display_name="Inspect Operator",
            description="Inspect Ascend operator layout and build context",
            prompt_template="",
            input_artifacts=[],
            output_artifacts=["operator_context.json", "operator_context.md"],
            tools=["inspect_ascend_operator"],
            allow_skip=False,
        )

    def execute(self, state) -> StageResult:
        ctx = ToolContext(cwd=str(state.project_root), auto_approve=True)
        params = {
            "project_root": str(state.project_root),
            "op_path": state.op_path or state.target,
            "soc_hint": state.soc,
        }
        result = self.tool_runner.execute("inspect_ascend_operator", params, ctx)
        if not result.ok:
            return StageResult(False, {}, error=result.error or "inspect_ascend_operator failed")

        context = _json_load(result.output, {})
        if not context:
            return StageResult(False, {}, error="inspect_ascend_operator returned empty context")

        state.workflow_kind = "ascend_ut"
        state.op_path = context.get("op_path", state.op_path)
        state.repo_name = context.get("repo_name", state.repo_name)
        state.category = context.get("category", state.category)
        state.op_name = context.get("op_name", state.op_name or state.op)
        state.generation_mode = context.get("generation_mode", "ut_generate")
        state.coverage_mode = context.get("coverage_mode", "single_run")
        state.enabled_layers = context.get("enabled_layers", [])
        state.target = context.get("op_path", state.target)
        state.target_slug = state.op_name

        provider = self._skill_provider(state)
        guidance = provider.get_stage_packet(
            "understand_function",
            generation_mode=context.get("generation_mode", "ut_generate"),
        )
        lines = [
            f"# Ascend operator context - {context.get('op_path', state.target)}",
            "",
            "## Mode",
            f"- Generation mode: `{context.get('generation_mode', 'unknown')}`",
            f"- Coverage mode: `{context.get('coverage_mode', 'unknown')}`",
            f"- Existing UT detected: `{context.get('existing_ut_detected', False)}`",
            f"- Compare scope: {', '.join(context.get('compare_scope', [])) or 'none'}",
            "",
            "## Layout",
            f"- Repo: `{context.get('repo_name', '')}`",
            f"- Category: `{context.get('category', '')}`",
            f"- Operator: `{context.get('op_name', '')}`",
            f"- Operator dir: `{context.get('operator_dir', '')}`",
            "",
            "## Enabled Layers",
        ]
        for layer in context.get("enabled_layers", []):
            lines.append(f"- `{layer}`")
        if context.get("excluded_layers"):
            lines.extend(["", "## Excluded Layers"])
            for layer in context.get("excluded_layers", []):
                lines.append(f"- `{layer}`")
        lines.extend(
            [
                "",
                "## Build",
                f"- Selected SoC: `{context.get('selected_soc', '')}`",
                f"- Supported SoCs: {', '.join(context.get('supported_socs', [])) or 'unknown'}",
                "",
                "### build.sh -h summary",
                "```text",
                context.get("build_help_summary", "build.sh -h unavailable"),
                "```",
                "",
                "## DType / Format",
                f"- DTypes: {', '.join(context.get('dtype_candidates', [])) or 'unknown'}",
                f"- Formats: {', '.join(context.get('format_candidates', [])) or 'unknown'}",
                "",
                "## Required Attributes",
                *(f"- `{name}`" for name in context.get("required_attrs", [])),
                "",
                "## Reference UT",
                *(f"- `{path}`" for path in context.get("reference_ut_paths", [])[:8]),
                "",
                "## Shared Harness",
                *(f"- `{path}`" for path in (context.get("reference_context", {}) or {}).get("shared_test_harness", [])[:8]),
                "",
                "## Shared Helpers",
                *(f"- `{path}`" for path in (context.get("reference_context", {}) or {}).get("shared_helpers", [])[:8]),
                "",
                "## Skill Guidance",
                "```text",
                guidance,
                "```",
            ]
        )
        markdown = "\n".join(line for line in lines if line is not None)

        state.save_artifact("operator_context.json", _render_json(context))
        state.save_artifact("operator_context.md", markdown)

        outputs = {
            "operator_context.json": _render_json(context),
            "operator_context.md": markdown,
        }

        if "op_host" not in context.get("enabled_layers", []):
            state.auto_stop_reason = "Current operator does not expose comparable op_host/op_api layers"
            return StageResult(
                True,
                outputs,
                message="No comparable `op_host`/`op_api` layers were found.",
                complete_workflow=True,
            )

        if _generation_mode(context) == "ut_enhance":
            return StageResult(
                True,
                outputs,
                message="Existing op_host/op_api UT detected. The workflow will enhance them in a snapshot and compare before/after coverage.",
            )

        return StageResult(True, outputs, message="Operator context collected for from-scratch UT generation")


class AscendRequirementsStage(AscendBaseStage):
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="generate_requirements",
            display_name="Generate Requirements",
            description="Build Ascend UT requirements from deterministic operator context",
            prompt_template="",
            input_artifacts=["operator_context.json"],
            output_artifacts=["requirements.md"],
            tools=[],
            allow_skip=False,
        )

    def execute(self, state) -> StageResult:
        context = _load_json_artifact(state, "operator_context.json")
        provider = self._skill_provider(state)
        layers = context.get("enabled_layers", [])
        generation_mode = _generation_mode(context)
        coverage_mode = _coverage_mode(context)
        guidance = provider.get_stage_packet(
            "generate_requirements",
            generation_mode=generation_mode,
        )

        lines = [
            f"# Ascend UT Requirements for {context.get('op_path', state.target)}",
            "",
            "## 1. Goals and Scope",
            "- Compare only `op_host` and `op_api`.",
            "- Exclude ST, op_graph, op_kernel, op_kernel_aicpu, and operator implementation changes.",
            "",
            "## 1.1 Mode Selection",
            f"- generation_mode: `{generation_mode}`",
            f"- coverage_mode: `{coverage_mode}`",
        ]

        if generation_mode == "ut_generate":
            lines.extend(
                [
                    "- No current-operator `op_host/op_api` UT are available, so Stage 4 should generate tests from scratch.",
                    "- It is valid to reference shared test harness files, common helpers, and similar operators from other directories.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "- Existing current-operator `op_host/op_api` UT are available and should be used as style and framework references.",
                    "- Stage 4 should enhance coverage by adding companion tests in an isolated snapshot rather than rewriting the original repository.",
                    "- Before/after coverage comparison uses the current operator's existing UT as the baseline.",
                    "",
                ]
            )

        lines.extend(
            [
            "## 2. Inputs and Constraints",
            f"- Selected SoC: `{context.get('selected_soc', '')}`",
            f"- Candidate dtypes: {', '.join(context.get('dtype_candidates', [])) or 'unknown'}",
            f"- Candidate formats: {', '.join(context.get('format_candidates', [])) or 'unknown'}",
            f"- Required attrs from op_host: {', '.join(context.get('required_attrs', [])) or 'none detected'}",
            "",
            "## 3. Layer Responsibilities",
            ]
        )

        if "op_host" in layers:
            lines.extend(
                [
                    "### op_host",
                ]
            )
            if context.get("is_tiling_only"):
                tiling_lines = [
                    "- This operator is **tiling-only**: the `op_host/` source directory contains tiling code but NO registerable `InferShape` function.",
                    "- Skip infershape tests entirely. Testing `InfershapeContextPara` here will SIGSEGV at runtime (the InferShape function pointer is null).",
                ]
                if context.get("is_aclnn_exclude"):
                    tiling_lines.append("- The operator uses `aclnn_exclude` (composite/delegate pattern), so there is no standalone aclnn kernel, but the tiling function is still testable.")
                tiling_lines.extend([
                    "- Focus exclusively on `TilingContextFaker`/`TilingContextPara`-based tiling tests.",
                    "",
                ])
                lines.extend(tiling_lines)
            elif context.get("is_aclnn_exclude"):
                lines.extend(
                    [
                        "- This operator uses `aclnn_exclude` (composite/delegate pattern) — no independent tiling kernel.",
                        "- Skip tiling tests. Include only infershape coverage.",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "- Must include both tiling and infershape coverage.",
                        "- TDD order: invalid dtype/attr paths before happy-path tiling and shape inference.",
                        "- CompileInfo type, namespace, headers, and NodeAttrs must match inspection output.",
                        "",
                    ]
                )
        if "op_api" in layers:
            lines.extend(
                [
                    "### op_api",
                    "- Cover nullptr, invalid dtype, shape mismatch, and one valid path.",
                    "- Add out/inplace variants only when the repo exposes corresponding headers or symbols.",
                    "",
                ]
            )
        lines.extend(
            [
                "## 4. Build and Execution",
                "- Commands come from the Ascend profile templates and run through `build.sh`.",
            ]
        )
        if generation_mode == "ut_generate":
            lines.extend(
                [
                    "- Generated files may be created directly under the operator's `tests/ut/<layer>/` directories.",
                    "- `CMakeLists.txt` must be generated for every enabled comparable layer directory.",
                ]
            )
        else:
            lines.extend(
                [
                    "- Coverage enhancement must run in an isolated repo snapshot so the original repository remains untouched.",
                    "- Existing UT files may be read for reference, but all writes must target the snapshot.",
                    "- Prefer adding companion `*_attest.cpp` files and reuse existing `CMakeLists.txt` when it already picks up directory sources.",
                ]
            )
        lines.extend(
            [
                "",
                "## 5. Coverage and Priorities",
                "- Priority order: op_host infershape -> op_api."
                if context.get("is_aclnn_exclude")
                else "- Priority order: op_host tiling -> op_api."
                if context.get("is_tiling_only")
                else "- Priority order: op_host tiling -> op_host infershape -> op_api.",
            ]
        )
        if coverage_mode == "before_after_compare":
            lines.extend(
                [
                    "- Mandatory thresholds: overall line coverage >= 90% and per-layer line coverage >= 85%.",
                    "- Comparison target: after-enhancement coverage should be >= baseline coverage for each enabled comparable layer.",
                    "- When enhanced coverage is below threshold or below baseline, prefer filling deferred placeholders before inventing more files.",
                ]
            )
        else:
            lines.extend(
                [
                    "- Mandatory thresholds: overall line coverage >= 80% and per-layer line coverage >= 80%.",
                    "- When generated coverage is below threshold, prefer filling deferred placeholders before inventing more files.",
                ]
            )
        lines.extend(
            [
                "",
                "## 6. Skill Guidance",
                "```text",
                guidance,
                "```",
            ]
        )

        content = "\n".join(lines)
        state.save_artifact("requirements.md", content)
        return StageResult(True, {"requirements.md": content}, message="Ascend UT requirements generated")


class AscendTestPlanStage(AscendBaseStage):
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="design_test_plan",
            display_name="Design Test Plan",
            description="LLM-validated multi-file Ascend UT plan (with free exploration)",
            prompt_template="",
            input_artifacts=["operator_context.json", "requirements.md"],
            output_artifacts=["test_plan.md", "test_plan.json"],
            tools=["exec_command", "read_file"],
            allow_skip=False,
        )

    def _tool_schemas(self) -> List[Dict[str, Any]]:
        allowed = set(self.config.tools)
        return [
            schema
            for schema in self.tool_runner.registry.to_llm_schema()
            if schema["function"]["name"] in allowed
        ]

    def _build_plan_agent_prompt(
        self,
        state,
        context: Dict[str, Any],
        infra_ctx: Dict[str, Any],
        base_cases: Dict[str, Any],
    ) -> str:
        op = context.get("op_name", "?")
        layout = infra_ctx.get("op_host_layout", {})
        warnings = infra_ctx.get("warnings", [])
        pure_opapi = layout.get("is_pure_opapi", False)
        trackable = infra_ctx.get("trackable_sources", {})
        op_host_files = trackable.get("op_host", [])
        op_api_files = trackable.get("op_api", [])

        cases_json = json.dumps(
            {"cases": base_cases.get("cases", []), "count": len(base_cases.get("cases", []))},
            ensure_ascii=False, indent=2,
        )[:12000]
        warnings_md = "\n".join(f"- {w}" for w in warnings) if warnings else "(无警告)"

        infra_summary = json.dumps({
            "is_pure_opapi": pure_opapi,
            "has_real_host_code": layout.get("has_real_host_code", False),
            "has_op_api_subdir": layout.get("has_op_api_subdir", False),
            "op_host_cpp_files": op_host_files,
            "op_api_cpp_files": op_api_files,
        }, ensure_ascii=False, indent=2)

        extra_rule = ""
        if pure_opapi:
            extra_rule = (
                "⚠️ **CRITICAL**: op_host 是 pure_opapi 布局（只有 op_api/ 子目录，无 tiling/infershape .cpp）。\n"
                "   → 删除所有 case_type 包含 tiling_ 或 infershape_ 的用例\n"
                "   → 只保留 op_api 层的测试用例\n"
            )

        host_files_desc = ", ".join(op_host_files) if op_host_files else "(none)"
        api_files_desc = ", ".join(op_api_files) if op_api_files else "(none)"

        return f"""你是 Ascend C 算子和编译/覆盖率系统专家。你的任务是为 `{op}` 算子**验证并增强**一份自动生成的测试计划。

## 算子基本信息
- op_name: `{op}`
- op_dir: `{context.get('operator_dir', '')}`
- project_root: `{str(state.project_root)}`
- op_host 源文件: {host_files_desc}
- op_api 源文件: {api_files_desc}

## 基础设施摘要
```json
{infra_summary}
```

## 基础设施警告
{warnings_md}

## 你的工具
- `exec_command`: 执行 shell 命令探索源码
- `read_file`: 读取文件内容

## 三阶段探索（必须依次完成）

### 阶段 A — 代码结构理解
执行以下命令，**每条都要实际执行**，记录你看到的内容：
1. `ls -la {context.get('operator_dir', '')}/op_host/` 列出 op_host 的所有源文件
2. 对 op_host 下的每个 .cpp 文件，`head -50` 读取文件开头（了解函数签名和宏注册）
3. `grep -rn 'aclnn\\|OP_API_UT' {context.get('operator_dir', '')}/op_api/ 2>/dev/null | head -60` 查找 API 层函数签名
4. 如果 exist: `ls -la {context.get('operator_dir', '')}/op_host/op_api/` 检查嵌套 op_api
5. `cat {context.get('operator_dir', '')}/tests/ut/op_host/CMakeLists.txt` 读取已有的 op_host 测试 CMake（如有）
6. `cat {context.get('operator_dir', '')}/tests/ut/op_api/CMakeLists.txt` 读取已有的 op_api 测试 CMake（如有）
7. `find {context.get('operator_dir', '')}/tests/ -name '*.cpp' 2>/dev/null` 查找已有测试文件

### 阶段 B — 测试需求推导
阅读完代码后，分析：
- op_host 层的代码包含哪些可测试路径？（函数签名、分支条件、dtype 支持）
- op_api 层的代码包含哪些可测试路径？（aclnn 函数、workspace 验证、shape/dtype 检查）
- 哪些 test case type 与实际代码匹配？哪些不对应？
- 是否发现了确定性计划**未覆盖**的可测试行为？

### 阶段 C — 计划验证与增强
对比「确定性候选计划」（见下方）与阶段 A-B 的发现：
- **保留**：与实际代码匹配的 case（特别是 priority=High 的）
- **删除**：只删除对应代码确实不存在的 case（需在 validation_notes 说明）
- **增强**：添加 1-5 个新的 case（使用新的 CASE_NN），覆盖确定性计划遗漏的场景
- ⚠️ **禁止**：不得大幅删减用例至 20 个以下（除非源代码确实很少）
- ⚠️ **禁止**：不得删除 priority=High 的用例（它们覆盖最基本的代码路径）

{extra_rule}

## 确定性候选计划（待验证）
```json
{cases_json}
```

## 输出格式

完成三个阶段后，输出一个 ```json ... ``` 块：

```json
{{
  "cases": [...],
  "smoke_set": ["CASE_01", ...],
  "deferred_set": ["CASE_03", ...],
  "validation_notes": "简要说明：(1) 每个 High-priority case 是否与代码匹配 (2) 删除了哪些 case 及原因 (3) 新增了什么 case 及原因",
  "code_analysis_notes": "2-3 句话总结你对该算子代码结构的理解（函数签名、分支条件、dtype 支持）"
}}
```

每个新增的 case 必须包含：
```json
{{"block_id": "CASE_NN", "tc_id": "TC-NN", "file_id": "FILE_02_OP_HOST_CPP",
  "layer_id": "op_host", "case_type": "snake_case_name", "priority": "High|Medium|Low",
  "name": "one-line description", "inputs": {{...}}, "expected": [...],
  "origin": "plan_agent"}}
```

现在开始阶段 A 的探索。"""

    def _parse_agent_output(self, llm_text: str) -> Dict[str, Any]:
        """Extract the final ```json ... ``` block from LLM output, preferring the last one."""
        matches = list(re.finditer(r"```json\s*(\{[\s\S]*?\})\s*```", llm_text or ""))
        for m in reversed(matches):
            try:
                plan = json.loads(m.group(1))
                if isinstance(plan, dict):
                    return plan
            except json.JSONDecodeError:
                cleaned = re.sub(r",\s*([}\]])", r"\1", m.group(1))
                try:
                    plan = json.loads(cleaned)
                    if isinstance(plan, dict):
                        return plan
                except json.JSONDecodeError:
                    continue
        return {}

    def _run_plan_agent_session(
        self, state, base_cases: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run a free-exploration LLM session to validate/refine the base test plan."""
        context = _load_json_artifact(state, "operator_context.json", default={})
        infra_ctx = _load_json_artifact(state, "infrastructure_context.json", default={})
        project_root = Path(state.project_root)
        if not project_root.exists():
            return None

        prompt = self._build_plan_agent_prompt(state, context, infra_ctx, base_cases)
        messages = [{"role": "user", "content": prompt}]
        append_message(
            session_id=getattr(state, "workflow_id", "workflow"),
            role="user",
            content={
                "stage": self.config.name,
                "mode": "plan_agent",
                "prompt": prompt,
            },
            workspace=str(state.workspace),
            stage=self.config.name,
        )

        ctx = ToolContext(cwd=str(project_root), auto_approve=True)
        tool_schemas = self._tool_schemas()

        turn_limit = 50
        stagnation_limit = 5
        stagnation_streak = 0
        last_tool_sig = ""
        api429_retries = 0
        api429_max = 3

        for turn in range(turn_limit):
            response = None
            while response is None:
                try:
                    response = self.llm.chat(messages, tools=tool_schemas)
                except Exception as exc:
                    exc_str = str(exc)
                    if ("429" in exc_str or "Throttl" in exc_str or "rate" in exc_str.lower()) and api429_retries < api429_max:
                        api429_retries += 1
                        delay = min(30 * api429_retries, 90)
                        print(f"  ⚠️ Plan agent rate limited (turn {turn+1}), retry {api429_retries}/{api429_max} after {delay}s")
                        import time
                        time.sleep(delay)
                        continue
                    print(f"  ⚠️ Plan agent LLM call failed (turn {turn+1}): {exc}")
                    return None

            assistant_msg = {
                "role": "assistant",
                "content": response.content,
                "reasoning_content": getattr(response, "reasoning_content", "") or "",
            }
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)
            append_message(
                session_id=getattr(state, "workflow_id", "workflow"),
                role="assistant",
                content=assistant_msg,
                workspace=str(state.workspace),
                stage=self.config.name,
            )

            if not response.has_tool_calls():
                parsed = self._parse_agent_output(response.content or "")
                if parsed and parsed.get("cases"):
                    validated = self._validate_plan_agent_output(parsed, base_cases, infra_ctx)
                    if validated:
                        return validated
                    return parsed
                if turn < turn_limit - 3:
                    nudge = (
                        "你没有使用工具探索代码，或者输出的JSON格式不正确。"
                        "请：(1) 先使用 exec_command 和 read_file 工具探索 op_host/op_api 目录结构和源文件，"
                        "然后 (2) 在最后一轮输出 ```json ... ``` 格式的计划。"
                    )
                    messages.append({"role": "user", "content": nudge})
                    continue
                return None

            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    messages.append({
                        "role": "user",
                        "content": "Tool call JSON truncated. Split into smaller chunks.",
                    })
                    continue
                tool_result = self.tool_runner.execute(tool_name, tool_args, ctx)
                tool_output = (tool_result.output if tool_result.ok else (tool_result.error or ""))
                if len(tool_output) > 8000:
                    tool_output = tool_output[:8000] + "\n... (truncated by framework)"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_output,
                })
                append_message(
                    session_id=getattr(state, "workflow_id", "workflow"),
                    role="tool",
                    content={
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "output": tool_output[:2000],
                        "ok": tool_result.ok,
                    },
                    workspace=str(state.workspace),
                    stage=self.config.name,
                )

            current_sig = json.dumps(
                [{"fn": tc["function"]["name"], "args": tc["function"]["arguments"]}
                 for tc in (response.tool_calls or [])],
                sort_keys=True,
            )
            if current_sig == last_tool_sig:
                stagnation_streak += 1
            else:
                stagnation_streak = 0
            last_tool_sig = current_sig
            if stagnation_streak >= stagnation_limit:
                print(
                    f"\n  ⚠️ Plan agent stagnation at turn {turn+1} "
                    f"({stagnation_limit} rounds); exiting early"
                )
                return None

            if len(messages) > 150:
                messages = _compress_old_messages(messages, keep_recent=50)

        return None

    def _validate_plan_agent_output(
        self,
        agent_plan: Dict[str, Any],
        base_cases: Dict[str, Any],
        infra_ctx: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Structurally validate the plan agent output.

        - Must preserve all High-priority cases from base
        - Must not blindly truncate below minimum
        - Returns validated plan or None if output is structurally invalid
        """
        agent_cases = agent_plan.get("cases", [])
        base_case_list = base_cases.get("cases", [])
        if not agent_cases:
            return None

        high_priority_base_ids = {
            c["block_id"] for c in base_case_list if c.get("priority") == "High"
        }
        agent_block_ids = {c.get("block_id", "") for c in agent_cases}

        missing_high = high_priority_base_ids - agent_block_ids
        if missing_high and len(missing_high) > len(high_priority_base_ids) * 0.3:
            print(
                f"  ⚠️ Plan agent dropped {len(missing_high)}/{len(high_priority_base_ids)} "
                f"High-priority cases; restoring from base plan"
            )
            base_map = {c["block_id"]: c for c in base_case_list}
            for bid in missing_high:
                if bid in base_map:
                    agent_cases.insert(0, base_map[bid])
            agent_plan["cases"] = agent_cases
            smoke = list(agent_plan.get("smoke_set", []))
            for bid in missing_high:
                if bid not in smoke:
                    smoke.insert(0, bid)
            agent_plan["smoke_set"] = smoke

        min_cases = max(12, len(base_case_list) // 3)
        pure_opapi = (infra_ctx.get("op_host_layout", {}) or {}).get("is_pure_opapi", False)
        if pure_opapi:
            min_cases = max(6, min_cases // 2)
        if len(agent_cases) < min_cases:
            print(
                f"  ⚠️ Plan agent over-truncated to {len(agent_cases)} cases "
                f"(min {min_cases}); restoring from base plan"
            )
            base_map = {c["block_id"]: c for c in base_case_list}
            for c in base_case_list:
                bid = c.get("block_id", "")
                if bid and bid not in agent_block_ids:
                    agent_cases.append(base_map[bid])
                    agent_block_ids.add(bid)
            agent_plan["cases"] = agent_cases
            smoke = list(agent_plan.get("smoke_set", []))
            deferred = list(agent_plan.get("deferred_set", []))
            for c in base_case_list:
                bid = c.get("block_id", "")
                if bid not in smoke and bid not in deferred:
                    (smoke if c.get("priority") == "High" else deferred).append(bid)
            agent_plan["smoke_set"] = smoke
            agent_plan["deferred_set"] = deferred

        return agent_plan

    def _build_cases(self, state, context: Dict[str, Any]) -> Dict[str, Any]:
        files = context.get("suggested_files", [])
        cpp_files = [entry for entry in files if entry.get("kind") == "cpp"]
        cases: List[Dict[str, Any]] = []
        smoke_set: List[str] = []
        deferred_set: List[str] = []
        next_case = 1

        def add_case(file_id: str, layer_id: str, case_type: str, priority: str, name: str, inputs: Dict[str, Any], expected: List[str]) -> None:
            nonlocal next_case
            block_id = f"CASE_{next_case:02d}"
            next_case += 1
            case = {
                "tc_id": f"TC-{next_case - 1:02d}",
                "block_id": block_id,
                "file_id": file_id,
                "layer_id": layer_id,
                "case_type": case_type,
                "priority": priority,
                "name": name,
                "inputs": inputs,
                "expected": expected,
                "depends_on": [],
                "test_name_hint": f"{block_id}_{case_type}",
            }
            cases.append(case)
            if priority == "High":
                smoke_set.append(block_id)
            else:
                deferred_set.append(block_id)

        file_lookup = {entry["path"]: entry for entry in cpp_files}
        op_name = context.get("op_name", state.op)
        category = context.get("category", state.category)
        supported_dtypes = context.get("supported_dtypes") or context.get("dtype_candidates") or []
        supported_formats = context.get("supported_formats") or context.get("format_candidates") or []

        def _find_path(layer_id: str, token: str) -> Optional[str]:
            for path, entry in file_lookup.items():
                if entry.get("layer_id") != layer_id:
                    continue
                if token in path:
                    return path
            return None

        tiling_path = _find_path("op_host", "tiling")
        infershape_path = _find_path("op_host", "infershape")
        api_path = _find_path("op_api", f"test_aclnn_{op_name}")

        if tiling_path and (not context.get("is_aclnn_exclude") or context.get("is_tiling_only")):
            file_id = file_lookup[tiling_path]["file_id"]
            add_case(file_id, "op_host", "tiling_invalid_dtype", "High", "unsupported dtype fails", {"dtype": "invalid_or_unsupported"}, ["GRAPH_FAILED"])
            add_case(file_id, "op_host", "tiling_valid_smoke", "High", "basic tiling success", {"dtype": "primary_supported_dtype"}, ["GRAPH_SUCCESS", "tiling key set"])
            add_case(file_id, "op_host", "tiling_attr_variant", "Medium", "attr-dependent tiling path", {"attrs": context.get("required_attrs", [])[:2]}, ["attr-specific path exercised"])
            for dtype in supported_dtypes:
                add_case(file_id, "op_host", f"tiling_dtype_{dtype}", "Medium", f"tiling with {dtype}", {"dtype": dtype, "shape": "{128,256}"}, ["GRAPH_SUCCESS or dtype-specific tiling path"])
            add_case(file_id, "op_host", "tiling_scalar_inputs", "Medium", "scalar/tiny shape tiling", {"shape": "[1]"}, ["tiling fallback or success"])
            add_case(file_id, "op_host", "tiling_large_shape", "Medium", "large shape tiling", {"shape": "[1024, 1024]"}, ["tiling workspace computed"])
            add_case(file_id, "op_host", "tiling_zero_dim", "Low", "zero-dim / empty tensor tiling", {"shape": "[]"}, ["graceful handle or skip"])
            add_case(file_id, "op_host", "tiling_broadcast", "Low", "broadcast shape tiling", {"shape_a": "[1, 64]", "shape_b": "[32, 64]"}, ["tiling with broadcast"])
            for fmt in supported_formats[:3]:
                add_case(file_id, "op_host", f"tiling_format_{fmt}", "Low", f"tiling format={fmt}", {"format": fmt, "shape": "typical"}, ["format-specific tiling path"])
            add_case(file_id, "op_host", "tiling_dynamic_shape", "Low", "dynamic shape tiling", {"shape": "{-1}"}, ["dynamic shape tiling handled"])
            add_case(file_id, "op_host", "tiling_rank3", "Low", "3D tensor tiling", {"shape": "{8,16,32}"}, ["tiling for rank-3"])
            add_case(file_id, "op_host", "tiling_rank4", "Low", "4D tensor tiling", {"shape": "{1,3,16,16}"}, ["tiling for rank-4"])

        if infershape_path:
            file_id = file_lookup[infershape_path]["file_id"]
            add_case(file_id, "op_host", "infershape_basic", "High", "basic shape inference", {"shape": "primary example"}, ["GRAPH_SUCCESS", "expected output shape"])
            add_case(file_id, "op_host", "infershape_boundary", "Medium", "shape boundary variant", {"shape": "boundary"}, ["boundary shape handled"])
            for dtype in supported_dtypes[:max(6, len(supported_dtypes))]:
                add_case(file_id, "op_host", f"infershape_dtype_{dtype}", "Medium", f"infershape with {dtype}", {"dtype": dtype, "shape": "{2,4,8}"}, ["GRAPH_SUCCESS", "dtype preserved"])
            add_case(file_id, "op_host", "infershape_scalar", "Medium", "scalar shape inference", {"shape": "[1]"}, ["GRAPH_SUCCESS", "scalar output"])
            add_case(file_id, "op_host", "infershape_high_rank", "Low", "high-rank tensor inference", {"shape": "[2, 4, 8, 16, 32]"}, ["GRAPH_SUCCESS", "rank preserved"])
            add_case(file_id, "op_host", "infershape_0d", "Low", "zero-dim inference", {"shape": "[]"}, ["GRAPH_SUCCESS or error"])
            add_case(file_id, "op_host", "infershape_mismatch_rank", "High", "rank mismatch fails", {"shape": "mismatch_rank"}, ["GRAPH_FAILED"])
            add_case(file_id, "op_host", "infershape_null_input", "High", "null input descriptor", {"input": "null"}, ["GRAPH_FAILED"])
            for fmt in supported_formats[:3]:
                add_case(file_id, "op_host", f"infershape_format_{fmt}", "Low", f"infershape format={fmt}", {"format": fmt}, ["GRAPH_SUCCESS", "format preserved"])
            add_case(file_id, "op_host", "infershape_dynamic", "Medium", "dynamic shape inference", {"shape": "{-1}"}, ["dynamic shape handled"])
            add_case(file_id, "op_host", "infershape_rank3", "Low", "3D tensor inference", {"shape": "{4,8,16}"}, ["GRAPH_SUCCESS"])
            add_case(file_id, "op_host", "infershape_rank4", "Low", "4D tensor inference", {"shape": "{2,3,8,8}"}, ["GRAPH_SUCCESS"])
            add_case(file_id, "op_host", "infershape_rank6", "Low", "6D tensor inference", {"shape": "{2,2,2,4,4,4}"}, ["GRAPH_SUCCESS"])

        if api_path:
            file_id = file_lookup[api_path]["file_id"]
            add_case(file_id, "op_api", "api_valid", "High", "valid input smoke", {"dtype": "primary_supported_dtype"}, ["ACL_SUCCESS"])
            add_case(file_id, "op_api", "api_invalid_dtype", "Medium", "invalid dtype path", {"dtype": "unsupported"}, ["ACLNN_ERR_PARAM_INVALID"])
            add_case(file_id, "op_api", "api_shape_mismatch", "Low", "shape mismatch path", {"shape": "mismatch"}, ["error or failure code"])
            add_case(file_id, "op_api", "api_null_output", "High", "null output tensor — use OP_API_UT_EXPECT with valid INPUT and null OUTPUT, NEVER use INPUT(nullptr)", {"input": "valid_tensor", "output": "(aclTensor*)nullptr"}, ["ACLNN_ERR_PARAM_NULLPTR"])
            add_case(file_id, "op_api", "api_null_workspace", "High", "null workspace pointer", {"workspace": "nullptr"}, ["ACLNN_ERR_PARAM_NULLPTR or workspace_size=0"])
            for dtype in supported_dtypes[:max(6, len(supported_dtypes))]:
                add_case(file_id, "op_api", f"api_dtype_{dtype}", "Medium", f"execute with {dtype}", {"dtype": dtype, "shape": "{32,64}"}, ["ACL_SUCCESS", "result verified"])
            add_case(file_id, "op_api", "api_inplace", "Medium", "inplace variant", {"inplace": True, "dtype": "primary_supported_dtype"}, ["ACL_SUCCESS", "output == input"])
            add_case(file_id, "op_api", "api_scalar_input", "Medium", "scalar tensor execution", {"shape": "[1]"}, ["ACL_SUCCESS"])
            add_case(file_id, "op_api", "api_large_shape", "Low", "large shape execution", {"shape": "[512, 512]"}, ["ACL_SUCCESS", "correct result"])
            add_case(file_id, "op_api", "api_zero_dim", "Low", "zero-dim tensor execution", {"shape": "[]"}, ["ACL_SUCCESS or error"])
            add_case(file_id, "op_api", "api_broadcast", "Low", "broadcast shape execution", {"shape_a": "[1, 64]", "shape_b": "[32, 64]"}, ["ACL_SUCCESS", "broadcast result"])
            add_case(file_id, "op_api", "api_boundary_fp", "High", "fp boundary values FLT_MAX/FLT_MIN/DBL_MIN/subnormal", {"values": "FLT_MAX, FLT_MIN, DBL_MIN, subnormal, -FLT_MAX", "dtype": "DT_FLOAT"}, ["ACL_SUCCESS"])
            add_case(file_id, "op_api", "api_boundary_zero_one", "Medium", "zeros and ones inputs", {"values": "all_zeros, all_ones", "dtype": "DT_FLOAT16"}, ["ACL_SUCCESS"])
            add_case(file_id, "op_api", "api_boundary_int", "Medium", "int boundary: INT_MIN/INT_MAX/zero/negative", {"values": "INT_MIN, INT_MAX, 0, -1", "dtype": "DT_INT32"}, ["ACL_SUCCESS"])
            add_case(file_id, "op_api", "api_sign_combos", "Medium", "positive/negative/zero sign combinations", {"values": "all_pos, all_neg, mixed_sign, zeros"}, ["ACL_SUCCESS"])
            if len(supported_dtypes) >= 2:
                add_case(file_id, "op_api", "api_dtype_pair_a", "Medium", f"mixed input: {supported_dtypes[0]} x {supported_dtypes[1]}", {"dtype_a": supported_dtypes[0], "dtype_b": supported_dtypes[1]}, ["ACL_SUCCESS or type-promotion error"])
            if len(supported_dtypes) >= 3:
                add_case(file_id, "op_api", "api_dtype_pair_b", "Medium", f"mixed input: {supported_dtypes[0]} x {supported_dtypes[2]}", {"dtype_a": supported_dtypes[0], "dtype_b": supported_dtypes[2]}, ["ACL_SUCCESS or type-promotion error"])
            add_case(file_id, "op_api", "api_rank3", "Medium", "3D tensor execution", {"shape": "{4,8,16}", "dtype": "DT_FLOAT16" if supported_dtypes else "primary_supported_dtype"}, ["ACL_SUCCESS"])
            add_case(file_id, "op_api", "api_rank4", "Low", "4D tensor execution", {"shape": "{2,3,8,8}", "dtype": "DT_FLOAT16" if supported_dtypes else "primary_supported_dtype"}, ["ACL_SUCCESS"])
            add_case(file_id, "op_api", "api_rank5", "Low", "5D tensor execution", {"shape": "{2,2,4,4,4}", "dtype": "DT_FLOAT16" if supported_dtypes else "primary_supported_dtype"}, ["ACL_SUCCESS"])
            add_case(file_id, "op_api", "api_extreme_large", "Low", "very large tensor", {"shape": "{2,65536}"}, ["ACL_SUCCESS"])
            add_case(file_id, "op_api", "api_large_shape_b", "Low", "large 3D shape", {"shape": "{128,256,64}"}, ["ACL_SUCCESS"])
            add_case(file_id, "op_api", "api_broadcast_high_rank", "Low", "high-rank broadcast", {"shape_a": "{1,1,64}", "shape_b": "{4,8,64}"}, ["ACL_SUCCESS", "broadcast result"])

        return {
            "cases": cases,
            "smoke_set": smoke_set,
            "deferred_set": deferred_set,
        }

    def _llm_augment_cases(
        self,
        state,
        context: Dict[str, Any],
        augment_request: Dict[str, Any],
        existing_cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Hybrid planning: on a coverage plateau, ask the LLM to design targeted
        test cases aimed at the specific uncovered branches (from augment_request),
        instead of re-enumerating generic rule-based cases.

        Returns a list of NEW case dicts (same schema as _build_cases) with fresh
        CASE_NN / TC-NN ids continuing after the existing rule cases. On any failure
        it returns [] so the caller falls back to the deterministic plan.
        """
        target_layers = [str(l) for l in augment_request.get("layers", [])]
        uncovered = augment_request.get("uncovered", [])
        if not target_layers or not uncovered:
            return []

        # file_id/path map for the plateaued layers so the LLM references real files
        files = context.get("suggested_files", [])
        layer_files = [
            {"file_id": f.get("file_id"), "path": f.get("path"), "layer_id": f.get("layer_id")}
            for f in files
            if isinstance(f, dict)
            and f.get("kind") == "cpp"
            and str(f.get("layer_id")) in target_layers
        ]
        if not layer_files:
            return []

        # Next block/tc index continues after the rule-generated cases
        start_idx = len(existing_cases) + 1
        op_name = context.get("op_name", state.op_name or state.op)

        prompt = f"""You are a test requirement planner for Ascend C++ operator `{op_name}`.
Coverage has PLATEAUED for layers {target_layers}. Below are the exact uncovered source
lines (with sample line numbers). Design NEW, TARGETED test cases that would exercise those
specific branches — not generic dtype/shape enumeration.

## Uncovered code (per layer)
```json
{json.dumps(uncovered, ensure_ascii=False, indent=2)[:3500]}
```

## Files you may target (use these exact file_id / layer_id values)
```json
{json.dumps(layer_files, ensure_ascii=False, indent=2)[:1500]}
```

## Output
Return ONLY a JSON array (no prose) of 3-8 case objects. Each object MUST have:
{{"file_id": "<one of the file_ids above>", "layer_id": "<matching layer>",
  "case_type": "snake_case_short", "priority": "High|Medium|Low",
  "name": "one-line intent describing which uncovered branch it hits",
  "inputs": {{"key": "value"}}, "expected": ["assertion or status"]}}
Focus each case on a concrete uncovered branch above. Wrap the array in ```json ... ```.
"""
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}])
            text = getattr(resp, "content", "") or ""
        except Exception:
            return []

        raw = self._extract_json_array(text)
        if not isinstance(raw, list) or not raw:
            return []

        valid_file_ids = {str(f["file_id"]) for f in layer_files}
        fid_to_layer = {str(f["file_id"]): str(f["layer_id"]) for f in layer_files}
        new_cases: List[Dict[str, Any]] = []
        idx = start_idx
        for item in raw:
            if not isinstance(item, dict):
                continue
            fid = str(item.get("file_id", ""))
            if fid not in valid_file_ids:
                continue
            block_id = f"CASE_{idx:02d}"
            layer_id = fid_to_layer.get(fid, str(item.get("layer_id", "")))
            case_type = str(item.get("case_type") or "augment")[:40]
            new_cases.append({
                "tc_id": f"TC-{idx:02d}",
                "block_id": block_id,
                "file_id": fid,
                "layer_id": layer_id,
                "case_type": case_type,
                "priority": item.get("priority") if item.get("priority") in {"High", "Medium", "Low"} else "High",
                "name": str(item.get("name") or f"augment case for {layer_id}")[:200],
                "inputs": item.get("inputs") if isinstance(item.get("inputs"), dict) else {},
                "expected": item.get("expected") if isinstance(item.get("expected"), list) else [],
                "depends_on": [],
                "test_name_hint": f"{block_id}_{case_type}",
                "origin": "llm_augment",
            })
            idx += 1
        return new_cases

    @staticmethod
    def _extract_json_array(text: str) -> Any:
        """Extract a JSON array from LLM output (fenced ```json ... ``` or bare)."""
        m = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", text or "")
        if not m:
            m = re.search(r"(\[[\s\S]*\])", text or "")
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except Exception:
            return None

    def execute(self, state) -> StageResult:
        context = _load_json_artifact(state, "operator_context.json")
        provider = self._skill_provider(state)
        cfg = load_config().get("profiles", {}).get("ascend_ut", {})
        command_cfg = cfg.get("commands", {})
        generation_mode = _generation_mode(context)
        coverage_mode = _coverage_mode(context)
        guidance = provider.get_stage_packet("design_test_plan", generation_mode=generation_mode)

        # Phase 2: when the reviewer requested planner augmentation for plateaued
        # layers (replan loop), carry the directive into the emitted test plan so
        # the downstream generator focuses on the named layers' uncovered lines.
        prev_analysis = _load_json_artifact(state, "analysis_plan.json", default={})
        augment_request = None
        if int(getattr(state, "epoch_current", 1) or 1) > 1 and prev_analysis.get("augment_request"):
            augment_request = prev_analysis.get("augment_request")

        case_payload = self._build_cases(state, context)

        # Hybrid planning: on an augment (plateau) re-plan, ask the LLM to design
        # targeted cases for the uncovered branches and APPEND them to the rule
        # cases. Falls back to pure rule cases if the LLM call fails.
        llm_augment_count = 0
        if augment_request:
            try:
                new_cases = self._llm_augment_cases(
                    state, context, augment_request, case_payload["cases"]
                )
            except Exception:
                new_cases = []
            if new_cases:
                case_payload["cases"].extend(new_cases)
                for c in new_cases:
                    (case_payload["smoke_set"] if c["priority"] == "High"
                     else case_payload["deferred_set"]).append(c["block_id"])
                llm_augment_count = len(new_cases)

        # LLM-driven plan validation: free-exploration agent session
        # (epoch 1 only; for epoch >1 the agent would see stale context)
        agent_validation_notes = ""
        if int(getattr(state, "epoch_current", 1) or 1) == 1 and not augment_request:
            try:
                agent_plan = self._run_plan_agent_session(state, case_payload)
                if agent_plan and agent_plan.get("cases"):
                    orig_count = len(case_payload["cases"])
                    case_payload["cases"] = agent_plan["cases"]
                    if "smoke_set" in agent_plan:
                        case_payload["smoke_set"] = agent_plan["smoke_set"]
                    if "deferred_set" in agent_plan:
                        case_payload["deferred_set"] = agent_plan["deferred_set"]
                    agent_validation_notes = str(agent_plan.get("validation_notes", "")) or ""
                    new_count = len(case_payload["cases"])
                    print(
                        f"  ✅ Plan agent validated and refined test plan: "
                        f"{orig_count} cases → {new_count} cases"
                    )
                    if agent_validation_notes:
                        print(f"     Notes: {agent_validation_notes[:200]}")
                else:
                    print("  ℹ️ Plan agent session produced no usable output; using deterministic plan")
            except Exception as exc:
                print(f"  ⚠️ Plan agent session failed ({exc}); using deterministic plan")

        files = context.get("suggested_files", [])
        build_plan: Dict[str, Dict[str, str]] = {}
        combined_layers = []
        for layer in context.get("enabled_layers", []):
            compile_key = LAYER_COMMAND_KEYS.get(layer, f"compile_{layer}")
            if compile_key not in command_cfg:
                continue
            compile_cmd = command_cfg[compile_key].format(
                op_name=context.get("op_name", state.op_name or state.op),
                soc=context.get("selected_soc", state.soc),
                op_path=context.get("op_path", state.op_path),
                project_root=str(state.project_root),
                repo_name=context.get("repo_name", state.repo_name),
                category=context.get("category", state.category),
                layer=layer,
            )
            coverage_cmd = compile_cmd + str(command_cfg.get("compile_cov_suffix", " --cov"))
            build_plan[layer] = {
                "compile": compile_cmd,
                "coverage": coverage_cmd,
            }
            combined_layers.append(layer)

        if len(combined_layers) > 1:
            layer_flags = " ".join(f"--{LAYER_FLAG_NAMES.get(layer, layer)}" for layer in combined_layers)
            combined_base = f"bash build.sh -u {layer_flags} --ops='{context.get('op_name', state.op_name or state.op)}' --soc='{context.get('selected_soc', state.soc)}'"
            build_plan["_combined"] = {
                "combined_compile": combined_base,
                "combined_coverage": combined_base + " --cov",
            }

        plan = {
            "workflow_kind": "ascend_ut",
            "generation_mode": generation_mode,
            "coverage_mode": coverage_mode,
            "category": context.get("category", state.category),
            "op_name": context.get("op_name", state.op_name or state.op),
            "op_path": context.get("op_path", state.op_path),
            "selected_soc": context.get("selected_soc", state.soc),
            "enabled_layers": context.get("enabled_layers", []),
            "compare_scope": context.get("compare_scope", []),
            "baseline": {
                "project_root": str(state.project_root),
                "existing_ut_detected": context.get("existing_ut_detected", False),
                "existing_ut_by_layer": context.get("existing_ut_by_layer", {}),
            },
            "generated": {
                "project_root": str(_generated_repo_root(state)) if generation_mode == "ut_enhance" else str(state.project_root),
                "isolated_snapshot": generation_mode == "ut_enhance",
                "strategy": "companion_files" if generation_mode == "ut_enhance" else "from_scratch",
            },
            "files": files,
            "cases": case_payload["cases"],
            "smoke_set": case_payload["smoke_set"],
            "deferred_set": case_payload["deferred_set"],
            "build_plan": build_plan,
            "reference_context": context.get("reference_context", {}),
        }
        if agent_validation_notes:
            plan["agent_validation_notes"] = agent_validation_notes
            code_notes = agent_plan.get("code_analysis_notes", "") if agent_plan else ""
            if code_notes:
                plan["code_analysis_notes"] = str(code_notes)
        if augment_request:
            plan["augment_directive"] = augment_request

        md_lines = [
            f"# Ascend UT Test Plan - {context.get('op_path', state.target)}",
            "",
            "## 1. Strategy",
            f"- generation_mode: `{generation_mode}`",
            f"- coverage_mode: `{coverage_mode}`",
            "- First round fills only the smoke set; deferred placeholders remain for coverage-driven iterations.",
            "- Test names must embed BLOCK_ID for analysis-stage traceability.",
            "",
            "## 2. Generated Files",
        ]
        if generation_mode == "ut_generate":
            md_lines.insert(4, "- Multi-file generation with one CMake file per enabled comparable layer.")
        else:
            md_lines.insert(4, "- Coverage enhancement uses companion `*_attest.cpp` files inside an isolated snapshot.")
        for entry in _ordered_files(plan):
            md_lines.append(f"- `{entry['file_id']}` -> `{entry['path']}` ({entry['layer_id']}, {entry['kind']})")

        md_lines.extend(["", "## 3. Smoke Set"])
        for block_id in plan["smoke_set"]:
            md_lines.append(f"- `{block_id}`")
        md_lines.extend(["", "## 4. Deferred Set"])
        for block_id in plan["deferred_set"]:
            md_lines.append(f"- `{block_id}`")
        md_lines.extend(["", "## 5. Build Plan"])
        for layer, commands in build_plan.items():
            if layer.startswith("_"):
                continue
            compile_cmd = commands.get("compile")
            if compile_cmd:
                md_lines.append(f"- `{layer}` compile: `{compile_cmd}`")
            coverage_cmd = commands.get("coverage")
            if coverage_cmd:
                md_lines.append(f"- `{layer}` coverage: `{coverage_cmd}`")
        if coverage_mode == "before_after_compare":
            md_lines.extend(
                [
                    "",
                    "## 6. Baseline vs Enhanced",
                    f"- Baseline root: `{state.project_root}`",
                    f"- Generated root: `{_generated_repo_root(state)}`",
                ]
            )
        md_lines.extend(
            [
                "",
                "## 7. Skill Guidance",
                "```text",
                guidance,
                "```",
            ]
        )

        plan_md = "\n".join(md_lines)
        plan_json = _render_json(plan)

        state.save_artifact("test_plan.json", plan_json)
        state.save_artifact("test_plan.md", plan_md)
        augment_note = f" (+{llm_augment_count} LLM-augmented cases)" if llm_augment_count else ""
        return StageResult(
            True,
            {"test_plan.json": plan_json, "test_plan.md": plan_md},
            message=f"Generated multi-file Ascend test plan with {len(plan['files'])} files and {len(plan['cases'])} cases{augment_note}",
        )


_STUB_MAX_CASES_PER_FILE = 18


class AscendCodeGenStage(AscendBaseStage):
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="generate_code",
            display_name="Generate Code",
            description="Generate Ascend C++ UT files with semantic block markers",
            prompt_template="",
            input_artifacts=["operator_context.json", "requirements.md", "test_plan.json"],
            output_artifacts=["generation_manifest.json"],
            tools=["exec_command"],
            allow_skip=False,
        )

    def _prepare_generated_repo(self, state, context: Dict[str, Any]) -> Path:
        target_root = _generated_repo_root(state)
        source_root = Path(state.project_root)
        source_git = source_root / ".git"
        target_git = target_root / ".git"
        if target_root.exists():
            if _is_enhance_mode(context) and source_git.exists() and not target_git.exists():
                shutil.rmtree(target_root)
            else:
                _patch_build_sh_for_isolation(target_root)
                return target_root

        ignore = shutil.ignore_patterns(
            ".attest",
            "__pycache__",
            ".pytest_cache",
            "build",
            "*.pyc",
        )
        shutil.copytree(source_root, target_root, ignore=ignore)
        if _is_enhance_mode(context) and source_git.exists() and not target_git.exists():
            raise RuntimeError(f"Generated snapshot is missing git metadata: {target_git}")

        _patch_build_sh_for_isolation(target_root)
        return target_root

    def _find_source_test_file(self, project_root: Path, attest_path: Path, op_name: str) -> Optional[Path]:
        """Find corresponding source test file for the operator.
        
        For operator tests (op_api/op_host), if _attest.cpp doesn't exist,
        look for source file like test_<op>.cpp or test_<op>.cpp.bak.
        """
        attest_dir = attest_path.parent
        
        # Try common patterns
        patterns = [
            f"test_{op_name}.cpp",
            f"test_{op_name}.cpp.bak",
            f"test_aclnn_{op_name}.cpp",
            f"test_aclnn_{op_name}.cpp.bak",
        ]
        
        for pattern in patterns:
            source_path = attest_dir / pattern
            if source_path.exists():
                return source_path

        if attest_dir.exists():
            candidates = sorted([
                f for f in attest_dir.glob("test_*.cpp")
                if not f.name.endswith("_attest.cpp")
            ])
            if candidates:
                return candidates[0]

        return None

    def _find_aclnn_header(self, project_root: Path, op_name: str) -> Optional[str]:
        """Find the aclnn header relative include path for the operator."""
        math_dir = project_root / "math" / op_name
        if not math_dir.exists():
            return None
        headers = sorted(math_dir.rglob("aclnn_*.h"))
        if not headers:
            return None
        best = None
        for h in headers:
            if op_name.replace("_", "") in h.stem.replace("_", ""):
                best = h
                break
        if best is None:
            best = headers[0]
        rel = best.relative_to(math_dir)
        depth = 3
        return "/".join([".."] * depth + list(rel.parts))

    def _scan_aclnn_functions(self, project_root: Path, op_name: str) -> Tuple[str, str]:
        """Scan all aclnn_*.h headers under math/<op_name>/ and extract real function names.

        Returns (main_aclnn_func, header_relative_to_math_dir) or ("", "") if none found.
        The returned path is relative to math/<op_name>/ (e.g. "op_api/aclnn_bitwiseand.h").
        """
        math_dir = project_root / "math" / op_name
        if not math_dir.exists():
            return "", ""
        headers = sorted(math_dir.rglob("aclnn_*.h"))
        if not headers:
            return "", ""

        candidates = []
        best_header = None

        for h in headers:
            try:
                text = h.read_text(encoding="utf-8", errors="ignore")
                func_names = re.findall(r'\b(aclnn[A-Za-z0-9]+?)\s*\(', text)
                non_ws = [f for f in func_names if "GetWorkspaceSize" not in f and "Inplace" not in f]
                ws_funcs = [f for f in func_names if "GetWorkspaceSize" in f]

                main_func = non_ws[0] if non_ws else (
                    ws_funcs[0].replace("GetWorkspaceSize", "") if ws_funcs else None
                )
                if not main_func:
                    continue

                sig_match = re.search(
                    r'\b' + re.escape(main_func) + r'GetWorkspaceSize\s*\(([^)]+)\)',
                    text,
                )
                if sig_match:
                    sig = sig_match.group(1)
                    tensor_params = re.findall(r'\bconst\s+aclTensor\s*\*', sig)
                    if len(tensor_params) != 1:
                        continue

                rel_path = h.relative_to(math_dir)
                rel_path_str = "/".join(rel_path.parts)

                if op_name.replace("_", "") in h.stem.replace("_", ""):
                    candidates.insert(0, (main_func, rel_path_str))
                else:
                    candidates.append((main_func, rel_path_str))
            except Exception:
                continue

        if not candidates:
            return "", ""

        best_func, best_include = candidates[0]
        return best_func, best_include

    @staticmethod
    def _op_has_op_api_dir(project_root: Path, op_name: str) -> bool:
        """Check if operator has an op_api directory (aclnn-based API).
        Some aclnn_exclude operators nest op_api under op_host/op_api/."""
        return (project_root / "math" / op_name / "op_api").is_dir() or \
               (project_root / "math" / op_name / "op_host" / "op_api").is_dir()

    @staticmethod
    def _to_pascal_case(snake: str) -> str:
        return "".join(part.capitalize() for part in snake.split("_") if part)

    def _generate_cpp_boilerplate(self, layer_id: str, op_name: str, project_root: Path) -> str:
        """Plan B: generate C++ test boilerplate when no source file exists."""
        if layer_id == "op_api":
            if not self._op_has_op_api_dir(project_root, op_name):
                return ""
            real_func, header_rel = self._scan_aclnn_functions(project_root, op_name)
            if not real_func:
                return ""
            include_depth = 3
            inc_path = "/".join([".."] * include_depth + header_rel.split("/"))
            class_name = f"{op_name}_test"

            ws_func = f"{real_func}GetWorkspaceSize"
            math_dir = project_root / "math" / op_name
            tensor_input_count = 1
            has_scalar = False
            has_aclscalar = False
            for h in sorted(math_dir.rglob("aclnn_*.h")):
                try:
                    text = h.read_text(encoding="utf-8", errors="ignore")
                    sig_match = re.search(
                        r'\b' + re.escape(ws_func) + r'\s*\(([^)]+)\)', text
                    )
                    if not sig_match:
                        continue
                    sig = sig_match.group(1)
                    tensor_params = len(re.findall(r'\bconst\s+aclTensor\s*\*', sig))
                    has_int64 = bool(re.findall(r'\bint64_t\b', sig))
                    has_aclscalar_param = bool(re.findall(r'\baclScalar\s*\*', sig))
                    if tensor_params > 0:
                        tensor_input_count = tensor_params
                        has_scalar = has_int64
                        has_aclscalar = has_aclscalar_param
                        break
                except Exception:
                    continue

            input_descs = "self_desc"
            input_decls = "  auto self_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);"
            for i in range(1, tensor_input_count):
                name = "other_desc" if i == 1 else f"extra{i}_desc"
                input_decls += f"\n  auto {name} = TensorDesc({{2, 3, 4, 5}}, ACL_FLOAT, ACL_FORMAT_ND);"
                input_descs += f", {name}"
            if has_aclscalar:
                input_decls += "\n  int64_t scalar_value = 1;\n  auto scalar_desc = ScalarDesc(scalar_value);"
                input_descs += ", scalar_desc"
            elif has_scalar:
                input_decls += "\n  int64_t scalar_value = 1;"
                input_descs += ", scalar_value"

            has_aclscalar_include = (
                '#include "op_api_ut_common/scalar_desc.h"\n' if has_aclscalar else ""
            )

            return (
                "#include <array>\n"
                "#include <vector>\n"
                '#include "gtest/gtest.h"\n'
                f'#include "{inc_path}"\n'
                '#include "op_api_ut_common/op_api_ut.h"\n'
                '#include "op_api_ut_common/tensor_desc.h"\n'
                f'{has_aclscalar_include}'
                "\n"
                "using namespace std;\n"
                "\n"
                f"class {class_name} : public testing::Test {{\n"
                " protected:\n"
                f"  static void SetUpTestCase() {{ cout << \"{op_name}_test SetUp\" << endl; }}\n"
                f"  static void TearDownTestCase() {{ cout << \"{op_name}_test TearDown\" << endl; }}\n"
                "};\n"
                "\n"
                f"TEST_F({class_name}, case_default_float32) {{\n"
                f"{input_decls}\n"
                "  auto out_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);\n"
                f"  auto ut = OP_API_UT({real_func}, INPUT({input_descs}), OUTPUT(out_desc));\n"
                "  uint64_t workspace_size = 0;\n"
                "  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);\n"
                "  EXPECT_EQ(aclRet, ACL_SUCCESS);\n"
                "}\n"
            )
        elif layer_id == "op_host":
            pascal = self._to_pascal_case(op_name)
            class_name = f"{pascal}InferShape"
            return (
                "#include <gtest/gtest.h>\n"
                "#include <iostream>\n"
                '#include "infershape_context_faker.h"\n'
                '#include "base/registry/op_impl_space_registry_v2.h"\n'
                "\n"
                f"class {class_name} : public testing::Test {{\n"
                " protected:\n"
                f"  static void SetUpTestCase() {{ std::cout << \"{class_name} SetUp\" << std::endl; }}\n"
                f"  static void TearDownTestCase() {{ std::cout << \"{class_name} TearDown\" << std::endl; }}\n"
                "};\n"
                "\n"
                "static std::vector<int64_t> ToVector(const gert::Shape& shape) {{\n"
                "  size_t n = shape.GetDimNum();\n"
                "  std::vector<int64_t> v(n, 0);\n"
                "  for (size_t i = 0; i < n; i++) v[i] = shape.GetDim(i);\n"
                "  return v;\n"
                "}\n"
            )
        return ""

    @staticmethod
    def _case_stub_lines(block_id: str, case_type: str, comment_style: str) -> List[str]:
        safe_type = re.sub(r"[^A-Za-z0-9_]", "_", str(case_type))[:40] or "stub"
        test_name = f"{block_id}_{safe_type}"
        return [
            start_marker(block_id, comment_style),
            f"// STUB: replace this block with real test for case_type={case_type}",
            f"TEST(AttestStubs, {test_name}) {{",
            f'    GTEST_SKIP() << "STUB — case_type=\\"{case_type}\\". Replace with real test body.";',
            f"}}",
            end_marker(block_id, comment_style),
        ]

    def _ensure_skeleton(self, project_root: Path, file_entry: Dict[str, Any], file_cases: List[Dict[str, Any]]) -> bool:
        path = project_root / str(file_entry["path"])
        comment_style = str(file_entry.get("comment_style") or detect_comment_style(path))
        is_cmake = str(file_entry.get("kind")) == "cmake"
        layer_id = str(file_entry.get("layer_id", ""))

        # Skip cmake files that don't exist yet (nothing to wrap)
        if is_cmake and not path.exists():
            return False

        # Plan A: For non-cmake operator tests, if _attest.cpp doesn't exist, copy source file first
        is_op_test = not is_cmake and layer_id in ("op_api", "op_host") and str(file_entry.get("kind", "")) in ("test_file", "cpp") and path.name.endswith("_attest.cpp")
        
        if is_op_test and not path.exists():
            op_name = str(file_entry.get("op_name", ""))
            op_path = str(file_entry.get("op_path", ""))
            if op_path and "/" in op_path:
                op_name = op_path.split("/")[-1]
            
            if op_name:
                source_file = self._find_source_test_file(project_root, path, op_name)
                if source_file:
                    print(f"📋 Found source file: {source_file.name}")
                    shutil.copy2(source_file, path)
                    print(f"📋 Copied source file to: {path.name}")
                    bak_file = source_file.with_suffix(source_file.suffix + ".bak")
                    if source_file != path and source_file.exists() and not bak_file.exists():
                        source_file.rename(bak_file)
                        print(f"📋 Renamed source to: {bak_file.name}")
                    return self._ensure_skeleton(project_root, file_entry, file_cases)

        if path.exists():
            existing_blocks = build_block_entries(path)
            if existing_blocks:
                return False
            original = path.read_text(encoding="utf-8")
            if not is_cmake:
                # Non-cmake file: wrap existing content as HEADER block
                lines = [
                    start_marker("HEADER", comment_style),
                    original.rstrip(),
                    end_marker("HEADER", comment_style),
                ]
                for case in file_cases:
                    lines.append(placeholder_marker(str(case["block_id"]), comment_style))
                if layer_id == "op_host":
                    lines.extend([
                        start_marker("FOOTER", comment_style),
                        f"{comment_style} TODO: The CMake registration below should be in CMakeLists.txt FOOTER, not here",
                        f"{comment_style} if(UT_TEST_ALL OR OP_HOST_UT)",
                        f"{comment_style}     add_modules_ut_sources(UT_NAME ${{OP_INFERSHAPE_MODULE_NAME}} MODE PRIVATE DIR ${{CMAKE_CURRENT_SOURCE_DIR}})",
                        f"{comment_style}     add_modules_ut_sources(UT_NAME ${{OP_TILING_MODULE_NAME}} MODE PRIVATE DIR ${{CMAKE_CURRENT_SOURCE_DIR}})",
                        f"{comment_style} endif()",
                        end_marker("FOOTER", comment_style),
                    ])
                else:
                    lines.append(start_marker("FOOTER", comment_style))
                    lines.append(end_marker("FOOTER", comment_style))
                content = "\n".join(lines) + "\n"
                ctx = ToolContext(cwd=str(project_root), auto_approve=True)
                self.tool_runner.execute("write_file", {"path": str(file_entry["path"]), "content": content}, ctx)
                return True
            else:
                # Cmake file: wrap with HEADER + FOOTER (FOOTER needed for _attest.cpp registration)
                lines = [
                    start_marker("HEADER", comment_style),
                    original.rstrip(),
                    end_marker("HEADER", comment_style),
                ]
                if layer_id == "op_host":
                    lines.extend([
                        start_marker("FOOTER", comment_style),
                        f"if(UT_TEST_ALL OR OP_HOST_UT)",
                        f"    add_modules_ut_sources(UT_NAME ${{OP_INFERSHAPE_MODULE_NAME}} MODE PRIVATE DIR ${{CMAKE_CURRENT_SOURCE_DIR}})",
                        f"    add_modules_ut_sources(UT_NAME ${{OP_TILING_MODULE_NAME}} MODE PRIVATE DIR ${{CMAKE_CURRENT_SOURCE_DIR}})",
                        f"endif()",
                        end_marker("FOOTER", comment_style),
                    ])
                else:
                    lines.extend([
                        start_marker("FOOTER", comment_style),
                        end_marker("FOOTER", comment_style),
                    ])
                content = "\n".join(lines) + "\n"
                ctx = ToolContext(cwd=str(project_root), auto_approve=True)
                self.tool_runner.execute("write_file", {"path": str(file_entry["path"]), "content": content}, ctx)
                return True

        ensure_parent(path)
        op_name_b = str(file_entry.get("op_name", ""))
        op_path_b = str(file_entry.get("op_path", ""))
        if op_path_b and "/" in op_path_b:
            op_name_b = op_path_b.split("/")[-1]
        if not op_name_b:
            parts = str(path).split("/")
            for i, part in enumerate(parts):
                if part == "math" and i + 1 < len(parts):
                    op_name_b = parts[i + 1]
                    break
        boilerplate = ""
        if not is_cmake and layer_id in ("op_api", "op_host") and op_name_b:
            has_source = False
            if path.parent.exists():
                source_files = [f for f in path.parent.glob("*.cpp") if not f.name.endswith("_attest.cpp")]
                if source_files:
                    has_source = True
            if not has_source:
                boilerplate = self._generate_cpp_boilerplate(layer_id, op_name_b, project_root)
        is_attest_gen_here = (
            layer_id in ("op_api", "op_host")
            and path.name.endswith("_attest.cpp")
            and not is_cmake
        )
        use_stubs = is_attest_gen_here and len(file_cases) <= _STUB_MAX_CASES_PER_FILE
        if is_attest_gen_here and len(file_cases) > _STUB_MAX_CASES_PER_FILE:
            print(f"  V23: {path.name} has {len(file_cases)} cases (>{_STUB_MAX_CASES_PER_FILE}), disabling stub pre-fill")
        lines = [
            start_marker("HEADER", comment_style),
            boilerplate.rstrip() if boilerplate else "",
            end_marker("HEADER", comment_style),
        ]
        if use_stubs:
            for case in file_cases:
                block_id = str(case["block_id"])
                case_type = str(case.get("case_type") or case.get("name") or block_id)
                lines.extend(self._case_stub_lines(block_id, case_type, comment_style))
        else:
            for case in file_cases:
                lines.append(placeholder_marker(str(case["block_id"]), comment_style))
        lines.append(start_marker("FOOTER", comment_style))
        lines.append(end_marker("FOOTER", comment_style))
        content = "\n".join(lines) + "\n"
        ctx = ToolContext(cwd=str(project_root), auto_approve=True)
        self.tool_runner.execute("write_file", {"path": str(file_entry["path"]), "content": content}, ctx)
        if not is_cmake and layer_id in ("op_api", "op_host") and op_name_b:
            cmake_path = path.parent / "CMakeLists.txt"
            if not cmake_path.exists():
                cmake_path.parent.mkdir(parents=True, exist_ok=True)
                if layer_id == "op_api":
                    cmake_content = (
                        "if(UT_TEST_ALL OR OP_API_UT)\n"
                        "    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                        "endif()\n"
                    )
                else:
                    cmake_content = (
                        "if(UT_TEST_ALL OR OP_HOST_UT)\n"
                        "    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                        "    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                        "endif()\n"
                    )
                cmake_path.write_text(cmake_content, encoding="utf-8")
        return True

    def _select_target_blocks(self, project_root: Path, state, plan: Dict[str, Any], file_entry: Dict[str, Any], analysis_plan: Dict[str, Any], created: bool = False, uncovered_for_layer: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
        file_id = str(file_entry["file_id"])
        file_cases = _cases_by_file(plan).get(file_id, [])
        smoke_ids = set(_smoke_set(plan))
        targets: List[str] = []
        path = project_root / str(file_entry["path"])
        block_entries = {entry["block_id"]: entry for entry in build_block_entries(path)} if path.exists() else {}
        if created or not path.exists():
            targets.extend(["HEADER", "FOOTER"])
        elif not block_entries:
            targets.extend(["HEADER", "FOOTER"])
        else:
            for block_id in ("HEADER", "FOOTER"):
                if block_entries.get(block_id, {}).get("status") == "placeholder":
                    targets.append(block_id)

        file_has_failures = False
        if analysis_plan.get("failures"):
            for item in analysis_plan.get("failures", []):
                if not isinstance(item, dict):
                    continue
                if item.get("file_id") != file_id:
                    continue
                block_id = str(item.get("block_id") or "")
                if block_id and block_id not in targets:
                    targets.append(block_id)
                file_has_failures = True
            if file_has_failures:
                return targets

        if getattr(state, "epoch_current", 1) == 1:
            for case in file_cases:
                if case["block_id"] in smoke_ids:
                    targets.append(str(case["block_id"]))
            return list(dict.fromkeys(targets))

        deferred_ids = set(_deferred_set(plan))
        placeholder_count = 0
        max_blocks_per_epoch = plan.get("block_limit", 6)
        for case in file_cases:
            block_id = str(case["block_id"])
            if block_id not in deferred_ids:
                continue
            entry = block_entries.get(block_id)
            if entry and entry.get("status") == "placeholder":
                targets.append(block_id)
                placeholder_count += 1
                if placeholder_count >= max_blocks_per_epoch:
                    break

        layer_id = str(file_entry.get("layer_id", ""))
        if uncovered_for_layer and layer_id and layer_id in uncovered_for_layer:
            udata = uncovered_for_layer[layer_id]
            if udata.get("files") and udata.get("total_uncovered", 0) > 0:
                if "HEADER" not in targets:
                    targets.append("HEADER")

        return list(dict.fromkeys(targets))

    def _tool_schemas(self) -> List[Dict[str, Any]]:
        allowed = set(self.config.tools)
        return [
            schema
            for schema in self.tool_runner.registry.to_llm_schema()
            if schema["function"]["name"] in allowed
        ]

    def _validate_generated_code(self, project_root: Path, file_entry) -> Optional[List[str]]:
        violations: List[str] = []
        target_path = project_root / str(file_entry["path"])
        if not target_path.exists():
            return None
        content = target_path.read_text(encoding="utf-8", errors="replace")
        if "TestPrecision" in content:
            violations.append(
                f"{str(file_entry['path'])} contains 'TestPrecision' call. "
                "TestPrecision() invokes real device API and will segfault in UT. "
                "Replace ALL TestPrecision() calls with TestGetWorkspaceSize(&ws)."
            )
        if re.search(r"INPUT\s*\([^)]*nullptr[^)]*\)", content):
            violations.append(
                "Code passes nullptr to INPUT macro; this will cause segfault. "
                "IMPORTANT: INPUT() must ALWAYS receive a valid TensorDesc (e.g. `auto in = TensorDesc({128}, ACL_FLOAT); ... INPUT(in)`). "
                "For nullptr validation, pass nullptr to OUTPUT only: `OP_API_UT_EXPECT(aclnnXxx, INPUT(valid_in), OUTPUT((aclTensor*)nullptr))` returns ACLNN_ERR_PARAM_NULLPTR."
            )
        if re.search(r"(?<!\w)aclnn\w+\s*\(", content) and "OP_API_UT" not in content:
            violations.append(
                "Code calls aclnn* API directly without OP_API_UT macro. "
                "Use OP_API_UT(aclnnXxx, INPUT(...), OUTPUT(...)) instead."
            )
        if re.search(r"\bint\s+main\s*\(", content):
            violations.append(
                "Code defines main() function. "
                "This conflicts with test_op_api_main.cpp. Remove the main() definition."
            )
        block_entries = build_block_entries(target_path)
        case_entries = [e for e in block_entries if e["block_id"].startswith("CASE_")]
        if case_entries and all(e["status"] == "placeholder" for e in case_entries):
            violations.append(
                "All CASE blocks are empty (no test code generated). "
                "You MUST fill at least one CASE block with a working test. "
                "Start with CASE_01: write a simple test that calls the operator with basic inputs."
            )
        return violations if violations else None

    def _run_compile_check(self, project_root: Path, layer_id: str, plan: Dict[str, Any]) -> tuple[bool, str]:
        build_plan = plan.get("build_plan") or {}
        layer_cmd = build_plan.get(layer_id, {})
        compile_cmd = layer_cmd.get("compile", "") if isinstance(layer_cmd, dict) else ""
        if not compile_cmd:
            return True, ""
        ctx = ToolContext(cwd=str(project_root), auto_approve=True)
        result = self.tool_runner.execute("exec_command", {"cmd": compile_cmd}, ctx, source="framework")
        output = (result.output if result.output else result.error or "")
        return result.ok, output

    def _run_binary_check(self, project_root: Path, layer_id: str, plan: Dict[str, Any]) -> tuple[bool, str]:
        build_plan = plan.get("build_plan") or {}
        layer_cmd = build_plan.get(layer_id, {})
        compile_cmd = layer_cmd.get("compile", "") if isinstance(layer_cmd, dict) else ""
        if not compile_cmd:
            return True, ""
        run_cmd = f"{compile_cmd} 2>&1 | tail -200"
        ctx = ToolContext(cwd=str(project_root), auto_approve=True)
        result = self.tool_runner.execute("exec_command", {"cmd": run_cmd}, ctx, source="framework")
        output = (result.output if result.output else result.error or "")
        return result.ok, output

    def _run_repair_with_verification(
        self,
        state,
        project_root: Path,
        file_entry: Dict[str, Any],
        error_kind: str,
        error_text: str,
        target_blocks: List[str],
        plan: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        file_path = str(file_entry["path"])
        layer_id = str(file_entry.get("layer_id", ""))
        compile_cmd = ""
        if plan:
            build_plan_dict = plan.get("build_plan") or {}
            layer_cmds = build_plan_dict.get(layer_id, {}) if isinstance(build_plan_dict, dict) else {}
            compile_cmd = str(layer_cmds.get("compile", ""))
        recompile_hint = f"exec_command(cmd=\"{compile_cmd}\")" if compile_cmd else "the build command for this layer"
        rerun_hint = f"exec_command(cmd=\"{compile_cmd} 2>&1 | tail -100\")" if compile_cmd else "the build command for this layer (build.sh runs tests automatically)"
        repair_prompt = f"""The C++ file `{file_path}` has a **{error_kind}**. Fix it in a reflection loop.

## FULL ERROR OUTPUT (read carefully):
```
{error_text}
```

## REFLECTION PROCESS (follow strictly):

1. **Read this error carefully**. Identify the exact line, file, error message, and what caused it.
2. **Reason about the root cause**:
   - If compile error: what API did you call with wrong arguments? What type mismatch?
   - If runtime abort / assertion failure: what input value (dtype/shape/scalar) did you pass that violates the API's contract?
3. **Read the source** of the failing function via `cat`/`grep` to confirm its actual signature / valid input contract.
   Do NOT guess — verify.
4. **Fix the code** via `sed` or re-write the file. Make the MINIMAL correct change.
5. **Re-compile**: {recompile_hint}
6. If compile passes, **re-run tests**: {rerun_hint}
7. Repeat steps 1-6 until tests run cleanly without abort.

When done, output a brief summary of:
- what the actual root cause was
- what code change resolved it
- what lesson you learned for future blocks in this same file
"""
        return self._run_llm_session(state, repair_prompt, project_root)

    def _repair_file(
        self,
        state,
        project_root: Path,
        file_entry: Dict[str, Any],
        compile_error: str,
        target_blocks: List[str],
    ) -> StageResult:
        file_path = str(file_entry["path"])
        repair_prompt = f"""The C++ file `{file_path}` failed to compile. Fix the compilation errors.

Target blocks to fix: {', '.join(target_blocks)}

Compilation errors:
```
{compile_error[:3000]}
```

Rules:
1. You have ONLY `exec_command`. Use `cat`, `grep`, `sed`, heredocs, etc. for all file operations.
2. Read the file first to understand the current state, then fix the errors.
3. Focus on the specific error messages: type mismatches, missing includes, incorrect API usage, etc.
4. Do not modify blocks that are not in the target set.
5. Keep the fix minimal — only change what is needed to resolve the compilation errors.

Fix the file now."""
        return self._run_llm_session(state, repair_prompt, project_root)

    def _ensure_cmake_attest_registration(
        self,
        project_root: Path,
        file_entry: Dict[str, Any],
    ) -> None:
        file_path = str(file_entry.get("path", ""))
        if not file_path.endswith("_attest.cpp"):
            return
        layer_id = str(file_entry.get("layer_id", ""))
        cpp_name = Path(file_path).name
        cmake_path = project_root / str(Path(file_path).parent / "CMakeLists.txt")
        if not cmake_path.exists():
            cmake_path.parent.mkdir(parents=True, exist_ok=True)
            if layer_id == "op_api":
                content = (
                    "if(UT_TEST_ALL OR OP_API_UT)\n"
                    "    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                    f"    if(NOT TARGET ${{OP_API_MODULE_NAME}}_cases_obj)\n"
                    f"        add_library(${{OP_API_MODULE_NAME}}_cases_obj OBJECT)\n"
                    f"    endif()\n"
                    f"    target_sources(${{OP_API_MODULE_NAME}}_cases_obj PRIVATE {cpp_name})\n"
                    "endif()\n"
                )
            else:
                content = (
                    "if(UT_TEST_ALL OR OP_HOST_UT)\n"
                    "    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                    "    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                    f"    if(NOT TARGET ${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj)\n"
                    f"        add_library(${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj OBJECT)\n"
                    f"    endif()\n"
                    f"    target_sources(${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj PRIVATE {cpp_name})\n"
                    "endif()\n"
                )
            cmake_path.write_text(content, encoding="utf-8")
            return
        content = cmake_path.read_text(encoding="utf-8")
        content, sanitized = _sanitize_cmake_footer(content)
        if cpp_name in content and not sanitized:
            return
        lines = content.splitlines()
        insert_idx = None
        footer_end_marker = "# ==== BLOCK:FOOTER END ===="
        for i, line in enumerate(lines):
            if footer_end_marker in line:
                insert_idx = i
                break
        if insert_idx is None:
            return
        footer_start_idx = None
        for i in range(insert_idx - 1, -1, -1):
            if "# ==== BLOCK:FOOTER START ====" in lines[i]:
                footer_start_idx = i
                break
        if footer_start_idx is not None:
            filtered = [l for l in lines[footer_start_idx + 1 : insert_idx]
                        if "add_modules_ut_sources" not in l
                        and "add_library" not in l
                        and "set(OP_API_TEST_SOURCES" not in l
                        and "set(OP_HOST_TEST_SOURCES" not in l]
            lines = lines[: footer_start_idx + 1] + filtered + lines[insert_idx:]
            insert_idx = footer_start_idx + 1 + len(filtered)
        if layer_id == "op_api":
            reg_lines = [
                f"if(NOT TARGET ${{OP_API_MODULE_NAME}}_cases_obj)",
                f"    add_library(${{OP_API_MODULE_NAME}}_cases_obj OBJECT)",
                f"endif()",
                f"target_sources(${{OP_API_MODULE_NAME}}_cases_obj PRIVATE {cpp_name})",
            ]
        else:
            reg_lines = [
                f"if(UT_TEST_ALL OR OP_HOST_UT)",
                f"    if(NOT TARGET ${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj)",
                f"        add_library(${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj OBJECT)",
                f"    endif()",
                f"    target_sources(${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj PRIVATE {cpp_name})",
                f"endif()",
            ]
        new_lines = lines[:insert_idx] + reg_lines + lines[insert_idx:]
        cmake_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def _build_combined_layer_prompt(
        self,
        state,
        file_entries: List[Dict[str, Any]],
        plan: Dict[str, Any],
        project_root: Path,
        all_target_blocks: Dict[str, List[str]],
        cases_by_file: Dict[str, List[Dict[str, Any]]],
        slim_context: Dict[str, Any],
        analysis_plan: Dict[str, Any],
        uncovered_for_layer: Dict[str, Any],
        prev_error_context: str,
        case_inventory_context: str,
        generation_mode: str,
        provider,
        infra_context_snippet: str = "",
    ) -> str:
        """Build a unified prompt for processing all files of one layer in a single session."""
        if not file_entries:
            return ""

        layer_id = str(file_entries[0].get("layer_id", ""))
        build_plan_dict = plan.get("build_plan") or {}
        layer_cmds = build_plan_dict.get(layer_id, {}) if isinstance(build_plan_dict, dict) else {}
        compile_cmd_for_layer = str(layer_cmds.get("compile", ""))
        coverage_cmd_for_layer = str(layer_cmds.get("coverage", ""))

        # Build per-file sections
        file_sections: List[str] = []
        for fe in file_entries:
            file_id = str(fe["file_id"])
            target_blocks = all_target_blocks.get(file_id, [])
            file_cases = cases_by_file.get(file_id, [])
            block_index = build_block_index_json(project_root / str(fe["path"]))
            file_sections.append(
                f"### File: `{fe['path']}` (file_id={file_id}, layer={fe.get('layer_id')}, kind={fe.get('kind')})\n"
                f"Target blocks: {', '.join(target_blocks)}\n"
                f"Cases:\n```json\n{json.dumps(file_cases, ensure_ascii=False, indent=2)}\n```\n"
                f"Current block index:\n```json\n{block_index}\n```\n"
            )

        # Build uncovered section for this layer
        uncovered_section = ""
        epoch = int(getattr(state, "epoch_current", 1) or 1)
        if epoch > 1 and layer_id in uncovered_for_layer:
            udata = uncovered_for_layer[layer_id]
            files_info = udata.get("files", [])
            parts: List[str] = []
            for fi in files_info[:4]:
                src_path = fi.get("source", "")
                snippets = fi.get("snippets", [])
                if not snippets:
                    continue
                parts.append(f"\n### Source: {src_path}")
                for s in snippets[:5]:
                    parts.append(
                        f"\nLine {s.get('line')} is NOT covered:\n```cpp\n{s.get('context', '')}\n```"
                    )
            if parts:
                uncovered_section = (
                    f"\n## UNCOVERED CODE from previous epoch ({udata.get('total_uncovered', 0)} lines total)\n"
                    "Design tests that specifically trigger these uncovered branches.\n"
                    + "\n".join(parts)
                    + "\n"
                )

        unified_section = ""
        if compile_cmd_for_layer:
            unified_section = f"""
## UNIFIED GENERATE-COMPILE-TEST MODE

You are processing ALL files for the `{layer_id}` layer in a SINGLE session.
Process them in order. After completing ALL files, compile once and fix any errors.

Build command (compiles AND runs tests):
```
exec_command(cmd="{compile_cmd_for_layer}")
```

Coverage command (run after all blocks are complete):
```
exec_command(cmd="{coverage_cmd_for_layer}")
```

For each file:
1. Write all target blocks using `cat >` heredocs or `sed`.
2. After finishing ALL files, run the build command above.
3. If compile errors → read the error, identify root cause, read source if needed, fix, re-compile.
4. If runtime abort → read the error, find the bad input, fix, re-compile.
5. After clean run, check coverage and add targeted tests for uncovered lines.

NOTE: build.sh compiles AND runs tests automatically — no `--run` flag needed.
"""

        skill_packet = provider.get_stage_packet("generate_code", layer_id, generation_mode=generation_mode)
        few_shot = _get_few_shot_examples(layer_id)

        prompt = f"""You are generating Ascend C++ UT code for multiple files in the `{layer_id}` layer.

Operator context:
```json
{json.dumps(slim_context, ensure_ascii=False, indent=2)[:8000]}
```

{_build_ws_sig_section(layer_id, slim_context)}

Epoch: {getattr(state, 'epoch_current', 1)}/{getattr(state, 'epoch_total', 1)}

## Files to process (in order):

{"".join(file_sections)}

Analysis plan:
```json
{json.dumps(analysis_plan, ensure_ascii=False, indent=2)}
```

Rules:
1. You have ONLY `exec_command` as a tool. Use it for ALL operations: read files (`cat`, `head`, `less`), search (`grep`, `find`), write files (`cat > file << 'BLOCK_MARKER_EOF'` or `tee`), edit files (`sed`, `patch`), compile, run tests, and check coverage.
2. Fill only the listed target blocks for each file. Do not modify other blocks.
3. Keep every test name traceable to its BLOCK_ID.
4. For `*.cpp`, use `// ==== BLOCK:... ====`. For `CMakeLists.txt`, use `# ==== BLOCK:... ====`.
5. Use relative paths rooted at the current working directory.
6. After completing ALL files, compile and fix errors.
7. Share fixture classes and helper functions across files in this layer (define in the first file, reference in others).
8. Keep generated code compile-oriented and minimal.
9. DO NOT rewrite an existing, non-empty file from scratch. If a case block already contains working code, leave it untouched. Use `sed` or a careful heredoc that preserves existing content when writing a block. Overwriting a file with fewer TEST_F cases than it already has is forbidden.

{few_shot}

Skill packet:
```text
{skill_packet}
```
{infra_context_snippet}{unified_section}{uncovered_section}{case_inventory_context}{prev_error_context}
Complete all files now, then compile and verify."""
        return prompt

    def _run_shared_layer_session(
        self,
        state,
        file_entries: List[Dict[str, Any]],
        plan: Dict[str, Any],
        project_root: Path,
        all_target_blocks: Dict[str, List[str]],
        cases_by_file: Dict[str, List[Dict[str, Any]]],
        slim_context: Dict[str, Any],
        analysis_plan: Dict[str, Any],
        uncovered_for_layer: Dict[str, Any],
        prev_error_context: str,
        case_inventory_context: str,
        generation_mode: str,
        provider,
        infra_context_snippet: str = "",
    ) -> StageResult:
        """Run a single LLM session that processes all files of one layer together."""
        prompt = self._build_combined_layer_prompt(
            state, file_entries, plan, project_root,
            all_target_blocks, cases_by_file, slim_context, analysis_plan,
            uncovered_for_layer, prev_error_context, case_inventory_context,
            generation_mode, provider,
            infra_context_snippet=infra_context_snippet,
        )
        # Give extra turns proportional to the number of files
        turn_limit = min(240, 100 * len(file_entries))
        return self._run_llm_session(state, prompt, project_root, turn_limit=turn_limit)

    def _run_llm_session(self, state, prompt: str, project_root: Path, turn_limit: int = 70) -> StageResult:
        messages = [{"role": "user", "content": prompt}]
        append_message(
            session_id=getattr(state, "workflow_id", "workflow"),
            role="user",
            content={"stage": self.config.name, "prompt": prompt},
            workspace=str(state.workspace),
            stage=self.config.name,
        )
        ctx = ToolContext(cwd=str(project_root), auto_approve=True)
        tool_schemas = self._tool_schemas()

        STAGNATION_LIMIT = 8
        _stagnation_streak = 0
        _last_tool_calls_signature = ""

        _api400_retries = 0

        for turn in range(turn_limit):
            _made_progress = False
            _api400_response = None
            while _api400_response is None:
                try:
                    _api400_response = self.llm.chat(messages, tools=tool_schemas)
                except Exception as exc:
                    exc_str = str(exc)
                    if "400" in exc_str and _api400_retries < 3:
                        _api400_retries += 1
                        warning_msg = (
                            f"⚠️ API returned HTTP 400 (arguments too large). "
                            f"Retry {_api400_retries}/3. "
                            f"Please split your output into smaller chunks: use `cat >` heredocs for the first ~100 lines, "
                            f"then use `cat >>` or `sed` for remaining content in batches of ~100 lines each. "
                            f"Keep each `exec_command` under 5000 characters."
                        )
                        messages.append({"role": "user", "content": warning_msg})
                        _made_progress = True
                        break
                    return StageResult(False, {}, error=f"LLM call failed: {exc}")
            if _api400_response is None:
                continue
            response = _api400_response

            assistant_msg = {
                "role": "assistant",
                "content": response.content,
                "reasoning_content": getattr(response, "reasoning_content", "") or "",
            }
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)
            append_message(
                session_id=getattr(state, "workflow_id", "workflow"),
                role="assistant",
                content=assistant_msg,
                workspace=str(state.workspace),
                stage=self.config.name,
            )

            if not response.has_tool_calls():
                return StageResult(True, {}, message=response.content or "Generated file blocks")

            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    warning_msg = (
                        "⚠️ Your previous tool call was rejected — the arguments were truncated (invalid JSON). "
                        "Please split your output into smaller chunks: use `cat >` heredocs for the first ~100 lines, "
                        "then use `cat >>` or `sed` for remaining content in batches of ~100 lines each. "
                        "Keep each `exec_command` under 5000 characters."
                    )
                    messages.append({"role": "user", "content": warning_msg})
                    _made_progress = True
                    continue
                tool_result = self.tool_runner.execute(tool_name, tool_args, ctx)
                if tool_result.ok and tool_name in {
                    "exec_command", "write_file", "replace_in_file", "replace_block", "append_to_file",
                }:
                    _made_progress = True
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result.output if tool_result.ok else tool_result.error or "",
                }
                messages.append(tool_msg)
                append_message(
                    session_id=getattr(state, "workflow_id", "workflow"),
                    role="tool",
                    content=tool_msg,
                    workspace=str(state.workspace),
                    stage=self.config.name,
                )

            # Stagnation detection
            current_sig = json.dumps(
                [
                    {"fn": tc["function"]["name"], "args": tc["function"]["arguments"]}
                    for tc in (response.tool_calls or [])
                ],
                sort_keys=True,
            )
            if not _made_progress and current_sig == _last_tool_calls_signature:
                _stagnation_streak += 1
            else:
                _stagnation_streak = 0
            _last_tool_calls_signature = current_sig

            if _stagnation_streak >= STAGNATION_LIMIT:
                print(
                    f"\n  ⚠️  Early stop: {STAGNATION_LIMIT} consecutive iterations"
                    " without progress; exiting tool-calling loop"
                )
                return StageResult(
                    success=False,
                    outputs={},
                    error=(
                        f"Early stop: {STAGNATION_LIMIT} consecutive iterations"
                        " without file modifications"
                    ),
                )

            # Sliding-window context compression: keep prompt + last 60 turns
            if turn > 0 and turn % 100 == 0 and len(messages) > 125:
                messages = _compress_old_messages(messages, keep_recent=60)

        return StageResult(False, {}, error="Maximum tool-calling iterations reached")

    def execute(self, state) -> StageResult:
        plan = _load_json_artifact(state, "test_plan.json")
        context = _load_json_artifact(state, "operator_context.json")
        analysis_plan = _load_json_artifact(state, "analysis_plan.json", default=_default_analysis_plan())
        provider = self._skill_provider(state)
        cases_by_file = _cases_by_file(plan)
        project_root = _llm_project_root(state, context)
        generation_mode = _generation_mode(context)
        coverage_mode = _coverage_mode(context)
        if _is_enhance_mode(context):
            project_root = self._prepare_generated_repo(state, context)
        prompt_context = dict(context)
        prompt_context["generated_project_root"] = str(project_root)
        if generation_mode == "ut_generate":
            prompt_context = _sanitize_context_for_generation(prompt_context)
        else:
            prompt_context = _rewrite_context_for_project_root(prompt_context, project_root)

        slim_context = {
            "op_name": prompt_context.get("op_name"),
            "generation_mode": prompt_context.get("generation_mode"),
            "coverage_mode": prompt_context.get("coverage_mode"),
            "selected_soc": prompt_context.get("selected_soc"),
            "enabled_layers": prompt_context.get("enabled_layers"),
            "is_aclnn_exclude": prompt_context.get("is_aclnn_exclude"),
            "dtype_candidates": prompt_context.get("dtype_candidates", [])[:20],
            "format_candidates": prompt_context.get("format_candidates", [])[:20],
            "build_help_summary": prompt_context.get("build_help_summary", "")[:500],
            "build_commands": prompt_context.get("build_commands", {}),
            "workspace_signatures": prompt_context.get("workspace_signatures", []),
        }
        slim_context.update(
            _read_op_source_code(
                Path(prompt_context.get("operator_dir", "")),
                str(prompt_context.get("op_name", "")),
            )
        )
        slim_context = {k: v for k, v in slim_context.items() if v is not None}

        infra_context_snippet = ""
        try:
            infra_ctx = _load_json_artifact(state, "infrastructure_context.json", default={})
            if infra_ctx:
                layout = infra_ctx.get("op_host_layout", {})
                cmake_info = infra_ctx.get("cmake_info", {})
                warnings = infra_ctx.get("warnings", [])
                infra_context_snippet_parts = ["\n## Infrastructure Context (MUST FOLLOW — verified by build probe)\n"]
                if layout.get("is_pure_opapi"):
                    infra_context_snippet_parts.append(
                        "WARNING: op_host/ contains ONLY op_api/ nesting — no infershape/tiling .cpp files. "
                        "DO NOT generate op_host tests unless you fully understand this layout. "
                        "Focus on op_api tests."
                    )
                if cmake_info.get("cmake_style"):
                    infra_context_snippet_parts.append(f"CMake style: `{cmake_info['cmake_style']}`")
                if cmake_info.get("targets"):
                    infra_context_snippet_parts.append(f"CMake UT targets: `{json.dumps(cmake_info['targets'])}`")
                if cmake_info.get("test_register_macro"):
                    infra_context_snippet_parts.append(f"Test register macro: `{cmake_info['test_register_macro']}`")
                tc = cmake_info.get("test_cmake", {})
                if tc.get("op_host_op_api_exists"):
                    infra_context_snippet_parts.append(
                        "IMPORTANT: tests/ut/op_host/op_api/ exists — op_api tests should be generated "
                        "under tests/ut/op_host/op_api/, NOT under tests/ut/op_api/."
                    )
                if not tc.get("op_host_exists"):
                    infra_context_snippet_parts.append(
                        "tests/ut/op_host/CMakeLists.txt does NOT exist — framework will generate it. "
                        "Use the standard `add_modules_ut_sources` registration pattern."
                    )
                if not tc.get("op_api_exists"):
                    infra_context_snippet_parts.append(
                        "tests/ut/op_api/CMakeLists.txt does NOT exist — framework will generate it. "
                        "Use the standard `add_modules_ut_sources` registration pattern."
                    )
                for w in warnings:
                    infra_context_snippet_parts.append(f"⚠️ {w}")
                infra_context_snippet = "\n".join(infra_context_snippet_parts) + "\n"
                infra_slime = {
                    "op_host_layout": layout,
                    "cmake_style": cmake_info.get("cmake_style"),
                    "cmake_targets": cmake_info.get("targets"),
                }
                infra_slime = {k: v for k, v in infra_slime.items() if v}
                if infra_slime:
                    slim_context["infrastructure"] = infra_slime
        except Exception:
            pass

        uncovered_for_layer: Dict[str, Dict[str, Any]] = {}
        if int(getattr(state, "epoch_current", 1) or 1) > 1:
            try:
                uncovered_payload = _load_json_artifact(state, "uncovered_code.json", default={})
                for layer_data in uncovered_payload.get("layers", []):
                    layer = str(layer_data.get("layer", ""))
                    if layer:
                        uncovered_for_layer[layer] = {
                            "files": layer_data.get("files", []),
                            "total_uncovered": layer_data.get("total_uncovered", 0),
                        }
            except Exception:
                pass

        # P4: load previous-epoch error log and case inventory for cross-epoch context
        prev_error_context = ""
        case_inventory_context = ""
        layer_focus_context = ""
        if int(getattr(state, "epoch_current", 1) or 1) > 1:
            try:
                prev_error_log = state.load_artifact("prev_epoch_error_log.txt") or ""
                if prev_error_log.strip():
                    prev_error_context = (
                        "\n## Previous Epoch Test & Coverage Results\n"
                        "Below is a summary of test outcomes, coverage numbers, and uncovered line counts "
                        "from the previous epoch. Use this to avoid regressions and focus on remaining gaps.\n"
                        "```\n" + str(prev_error_log)[:4000] + "\n```\n"
                    )
            except Exception:
                pass
            try:
                inv = _load_json_artifact(state, "case_inventory.json", default={})
                blocks = inv.get("generated_blocks", [])
                if blocks:
                    bounded = [b["block_id"] for b in blocks if b.get("status") == "bounded"]
                    placeholder = [b["block_id"] for b in blocks if b.get("status") == "placeholder"]
                    placeholder_by_file: Dict[str, List[str]] = {}
                    for b in blocks:
                        if b.get("status") == "placeholder":
                            placeholder_by_file.setdefault(b.get("file_id", "?"), []).append(b["block_id"])
                    placeholder_detail = "; ".join(
                        f"{fid}: {', '.join(bids[:20])}"
                        for fid, bids in placeholder_by_file.items()
                    )
                    case_inventory_context = (
                        "\n## Previous Epoch Case Status\n"
                        f"Already bounded blocks ({len(bounded)} total, do NOT regenerate): {', '.join(bounded[:30]) or 'none'}\n"
                        f"Still placeholder ({len(placeholder)} total, FOCUS of this epoch): {', '.join(placeholder[:30]) or 'none'}\n"
                    )
                    if placeholder_detail:
                        case_inventory_context += f"Placeholder breakdown by file: {placeholder_detail}\n"
            except Exception:
                pass
            # Layer focus plan: read previous coverage to tell the LLM which layers
            # need work vs. which are already at threshold. Prevents the LLM from
            # wasting turns adding redundant cases to layers that already pass.
            try:
                cov = _load_json_artifact(state, "coverage_summary.json", default={})
                per_layer = cov.get("per_layer", {}) if isinstance(cov, dict) else {}
                threshold = cov.get("threshold", _layer_coverage_threshold(context))
                at_threshold_layers: List[str] = []
                below_threshold_layers: List[str] = []
                for layer_name, layer_meta in per_layer.items():
                    if not isinstance(layer_meta, dict):
                        continue
                    cov_val = layer_meta.get("line_coverage", 0.0)
                    try:
                        cov_pct = float(cov_val)
                    except (TypeError, ValueError):
                        continue
                    label = f"{layer_name} ({cov_pct}%)"
                    if cov_pct >= float(threshold):
                        at_threshold_layers.append(label)
                    else:
                        below_threshold_layers.append(label)
                if at_threshold_layers or below_threshold_layers:
                    parts = ["\n## Layer Focus Plan (CRITICAL — MUST FOLLOW)"]
                    if at_threshold_layers:
                        parts.append(
                            f"✅ Already at threshold — DO NOT add more cases to these layers: "
                            f"{', '.join(at_threshold_layers)}. "
                            f"Any case added here is WASTED effort."
                        )
                    if below_threshold_layers:
                        parts.append(
                            f"❌ Below threshold — ALLOCATE ALL turns and cases here: "
                            f"{', '.join(below_threshold_layers)}. "
                            f"For each placeholder case in these layers, fill with a realistic dtype+shape test. "
                            f"Consult `uncovered_code.json` for specific functions/lines to target."
                        )
                    parts.append(
                        "Strategy: skip op_host if ≥ threshold; if op_api is below, "
                        "focus 100% of generation effort on its placeholder cases and uncovered functions."
                    )
                    layer_focus_context = "\n".join(parts) + "\n"
            except Exception:
                pass
        # Prepend layer-focus to ensure the LLM sees this directive FIRST in the prompt
        if layer_focus_context:
            case_inventory_context = layer_focus_context + case_inventory_context

        manifest: List[Dict[str, Any]] = []
        cross_file_notes: List[str] = []

        # Check if shared-layer-session mode is enabled (Phase 2, off by default)
        _use_shared = load_config().get("profiles", {}).get("ascend_ut", {}).get(
            "enable_shared_layer_session", False
        )

        # First pass: ensure skeletons for all files; collect cmake entries
        files_needing_llm: List[Dict[str, Any]] = []
        for file_entry in _ordered_files(plan):
            file_id = str(file_entry["file_id"])
            file_cases = cases_by_file.get(file_id, [])
            created = self._ensure_skeleton(project_root, file_entry, file_cases)
            if created and str(file_entry.get("kind")) == "cmake":
                manifest.append({
                    "file_id": file_id,
                    "path": file_entry["path"],
                    "action": "updated",
                    "target_blocks": ["HEADER", "FOOTER"],
                })
                continue
            target_blocks = self._select_target_blocks(
                project_root, state, plan, file_entry, analysis_plan,
                created=created, uncovered_for_layer=uncovered_for_layer,
            )
            if not target_blocks:
                manifest.append({"file_id": file_id, "path": file_entry["path"], "action": "skip"})
                continue
            files_needing_llm.append({
                "entry": file_entry,
                "file_id": file_id,
                "target_blocks": target_blocks,
                "created": created,
            })

        # Group non-cmake files by layer
        files_by_layer: Dict[str, List[Dict[str, Any]]] = {}
        for item in files_needing_llm:
            layer = str(item["entry"].get("layer_id", ""))
            files_by_layer.setdefault(layer, []).append(item)

        # Process each layer group
        for layer_id_key in sorted(files_by_layer, key=lambda l: LAYER_ORDER.get(l, 99)):
            layer_items = files_by_layer[layer_id_key]

            if _use_shared and len(layer_items) > 1:
                # --- Shared session path (Phase 2) ---
                all_target_blocks = {item["file_id"]: item["target_blocks"] for item in layer_items}
                shared_result = self._run_shared_layer_session(
                    state=state,
                    file_entries=[item["entry"] for item in layer_items],
                    plan=plan,
                    project_root=project_root,
                    all_target_blocks=all_target_blocks,
                    cases_by_file=cases_by_file,
                    slim_context=slim_context,
                    analysis_plan=analysis_plan,
                    uncovered_for_layer=uncovered_for_layer,
                    prev_error_context=prev_error_context,
                    case_inventory_context=case_inventory_context,
                    generation_mode=generation_mode,
                    provider=provider,
                    infra_context_snippet=infra_context_snippet,
                )
                session_ok = shared_result.success

                # Post-session validation for each file in the group
                for item in layer_items:
                    file_entry = item["entry"]
                    file_id = item["file_id"]
                    target_blocks = item["target_blocks"]
                    layer_id = str(file_entry["layer_id"])

                    if str(file_entry.get("kind")) == "cmake":
                        self._ensure_cmake_attest_registration(project_root, file_entry)
                        manifest.append({
                            "file_id": file_id,
                            "path": file_entry["path"],
                            "action": "updated",
                            "target_blocks": target_blocks,
                        })
                        continue

                    code_violations = self._validate_generated_code(project_root, file_entry)
                    if code_violations:
                        violated_path = project_root / str(file_entry["path"])
                        quarantine = violated_path.with_suffix(violated_path.suffix + ".safety_violation.bak")
                        try:
                            shutil.move(str(violated_path), str(quarantine))
                        except Exception:
                            pass
                        cross_file_notes.append(
                            f"{file_id}: safety violation - {'; '.join(code_violations)[:200]}; skipping"
                        )
                        manifest.append({
                            "file_id": file_id,
                            "path": file_entry["path"],
                            "action": "error",
                            "reason": f"safety_violations: {'; '.join(code_violations)[:300]}",
                        })
                        continue

                    self._ensure_cmake_attest_registration(project_root, file_entry)

                    MAX_REPAIR_ROUNDS = 1
                    verified = False
                    for repair_round in range(MAX_REPAIR_ROUNDS):
                        compile_ok, compile_output = self._run_compile_check(project_root, layer_id, plan)
                        if not compile_ok:
                            error_summary = compile_output[:4000]
                            print(f"  ⚠️  Compile check failed for {file_id} (round {repair_round + 1}); "
                                  f"attempting LLM repair...")
                            repair_result = self._run_repair_with_verification(
                                state, project_root, file_entry,
                                error_kind="compile_error",
                                error_text=error_summary,
                                target_blocks=target_blocks,
                                plan=plan,
                            )
                            if not repair_result.success:
                                cross_file_notes.append(
                                    f"{file_id}: compile failed after repair round {repair_round + 1}"
                                )
                                break
                            continue
                        run_ok, run_output = self._run_binary_check(project_root, layer_id, plan)
                        abort_keywords = ("Assertion", "Aborted", "SIGSEGV", "core dumped",
                                          "terminate called", "Subprocess aborted")
                        if not run_ok or any(kw in run_output for kw in abort_keywords):
                            abort_snippet = run_output[:4000]
                            print(f"  ⚠️  Runtime abort/assertion failure for {file_id} (round {repair_round + 1}); "
                                  f"attempting in-session reflection fix...")
                            repair_result = self._run_repair_with_verification(
                                state, project_root, file_entry,
                                error_kind="runtime_abort",
                                error_text=abort_snippet,
                                target_blocks=target_blocks,
                                plan=plan,
                            )
                            if not repair_result.success:
                                cross_file_notes.append(
                                    f"{file_id}: runtime abort after repair round {repair_round + 1}"
                                )
                                break
                            continue
                        verified = True
                        break
                    else:
                        if not verified:
                            cross_file_notes.append(
                                f"{file_id}: still failing after {MAX_REPAIR_ROUNDS} repair rounds; continuing"
                            )

                    if session_ok:
                        cross_file_notes.append(f"{file_id}: completed successfully (shared session, {len(target_blocks)} blocks)")
                    manifest.append({
                        "file_id": file_id,
                        "path": file_entry["path"],
                        "action": "updated" if verified else "updated_unverified",
                        "target_blocks": target_blocks,
                    })

            else:
                # --- Original per-file session path (single file OR shared disabled) ---
                for item in layer_items:
                    file_entry = item["entry"]
                    file_id = item["file_id"]
                    target_blocks = item["target_blocks"]
                    layer_id = str(file_entry.get("layer_id", ""))

                    shared_harness_paths = _shared_harness_for_layer(prompt_context, layer_id)
                    shared_helper_paths = _shared_helpers_for_layer(prompt_context, layer_id)
                    operator_impl_paths = _operator_impl_for_layer(prompt_context, layer_id)
                    similar_example_paths = _similar_examples_for_layer(prompt_context, layer_id)
                    current_operator_ut_paths = _current_operator_ut_for_layer(prompt_context, layer_id)

                    existing_scenarios: List[str] = []
                    baseline_root = Path(state.project_root)
                    for ut_path_str in current_operator_ut_paths:
                        ut_path = baseline_root / ut_path_str
                        existing_scenarios.extend(_extract_existing_test_scenarios(ut_path))
                    if len(existing_scenarios) > 30:
                        existing_scenarios = existing_scenarios[:30]

                    if generation_mode == "ut_generate":
                        mode_rules = [
                            "9. Current-operator `op_host/op_api` UT do not exist in this mode. Generate from scratch and do not invent references to missing local UT files.",
                            "10. Use the listed shared harness, shared helpers, operator implementation files, and similar examples before doing broad recursive searches.",
                            "11. If you need to inspect a directory, use `ls` or `find` via `exec_command`. Do not call `cat` on directories.",
                            "12. Use relative paths rooted at the current working directory. Do not call any tool with an absolute path.",
                        ]
                        mode_guidance = [
                            "- `ut_generate`: no current-operator `op_host/op_api` UT exist; generate compile-oriented tests from scratch.",
                            "- Shared harness files and similar operators are the primary references for style and helper usage.",
                            "- DType name mapping: DT_FLOAT→ACL_FLOAT, DT_FLOAT16→ACL_FLOAT16, DT_BF16→ACL_BF16, "
                            "DT_INT32→ACL_INT32, DT_INT64→ACL_INT64, DT_DOUBLE→ACL_DOUBLE, DT_BOOL→ACL_BOOL, "
                            "DT_INT8→ACL_INT8, DT_INT16→ACL_INT16, DT_UINT8→ACL_UINT8.",
                            "- Each TEST_F block should cover multiple dtype+shape combinations for diversity (see Rule 16).",
                        ]
                        reference_paths = similar_example_paths
                    else:
                        mode_rules = [
                            "9. Current-operator `op_host/op_api` UT exist in this snapshot and may be read as references. Do not overwrite files outside the generated snapshot.",
                            "10. Prefer companion `*_attest.cpp` files and preserve the current operator's existing UT structure unless the target block explicitly requires otherwise.",
                            "11. Use the listed current-operator UT, shared harness, shared helpers, and operator implementation files before doing broad recursive searches.",
                            "12. Use relative paths rooted at the current working directory. Do not call any tool with an absolute path.",
                        ]
                        mode_guidance = [
                            "- `ut_enhance`: enhance coverage in the generated snapshot while keeping the original repository untouched.",
                            "- Current-operator UT references below are the primary style and harness references for this layer.",
                            "- CRITICAL: NEVER call `TestPrecision()` in generated tests. `TestPrecision()` invokes the real device kernel and will segfault in the UT environment. "
                            "Only call `TestGetWorkspaceSize(&ws)` which safely validates parameters and calculates workspace size without device execution.",
                            "- To maximize coverage, each TEST_F block MUST contain multiple `TestGetWorkspaceSize` calls covering: "
                            "(a) at least 3 different dtypes: ACL_FLOAT16, ACL_FLOAT, ACL_BF16, ACL_INT32, ACL_INT64 — use `aclCreateTensor` with each dtype, "
                            "(b) at least 3 different shapes: {1}, {32,64}, {2,3,8,8}, {128,256}, {65536}, "
                            "(c) boundary values: NaN, Inf, zeros, INT_MIN/INT_MAX, and negative values via appropriate data initialization, "
                            "(d) dtype promotion: mixed dtype inputs (e.g. FLOAT16 input + INT32 input) where the operator supports it.",
                            "- DType name mapping: DT_FLOAT→ACL_FLOAT, DT_FLOAT16→ACL_FLOAT16, DT_BF16→ACL_BF16, "
                            "DT_INT32→ACL_INT32, DT_INT64→ACL_INT64, DT_DOUBLE→ACL_DOUBLE, DT_BOOL→ACL_BOOL, "
                            "DT_INT8→ACL_INT8, DT_INT16→ACL_INT16, DT_UINT8→ACL_UINT8.",
                            "- Read existing baseline UT files to discover the correct pattern for calling `TestGetWorkspaceSize()` with proper input/output tensor setup.",
                        ]
                        reference_paths = list(dict.fromkeys(current_operator_ut_paths + similar_example_paths))

                    op_api_macro_rule = [
                        "13. For op_api layers: the `OP_API_UT` / `OP_API_UT_EXPECT` macros internally token-paste `GetWorkspaceSize` (and similar helpers) onto the first argument. Always pass a plain C function name like `aclnnRealDiv`, never a function pointer or variable. If the operator header only exposes function-pointer typedefs (e.g., `const aclTensor* (*RealDiv)(...)`), first read existing op_api UT files in the same directory via `cat` to discover the correct invocation pattern — do not guess.",
                        "14. Do NOT duplicate test scenarios already covered by existing baseline UT. Review the existing scenarios below and generate tests that target DIFFERENT dtype/format/shape combinations, different error paths, or uncovered functions listed in the analysis plan.",
                        "15. CRITICAL — CMakeLists.txt registration: When you generate a new `*_attest.cpp` file (FILE_02 for op_host, FILE_04 for op_api), you MUST also update the corresponding CMakeLists.txt FOOTER block to register the new file in the build system. For op_api: add the attest `.cpp` filename to the `OP_API_TEST_SOURCES` or `OP_API_MODULE_NAME_cases_obj` source list. For op_host: add the attest `.cpp` filename to `add_modules_ut_sources` or the appropriate source list. Read the existing CMakeLists.txt structure first and follow the same pattern. Without this registration, the test file will NOT be compiled and coverage will remain unchanged.",
                        "16. DIVERSITY REQUIREMENT: Each TEST_F in a block MUST use a distinct dtype+shape+value combination. "
                        "Cover as many different scenarios as possible within each block: "
                        "at least 3 different dtypes (e.g. ACL_FLOAT16, ACL_FLOAT, ACL_INT32, ACL_BF16, ACL_INT64), "
                        "at least 3 different shapes (e.g. {1}, {32,64}, {2,3,8,8}, {128,256}), "
                        "boundary values (zeros, ones, INT_MIN/INT_MAX, negative values, FLT_MIN/FLT_MAX), "
                        "and boundary tensor configurations (0D scalar, 1D, broadcast, high-rank 4D/5D). "
                        "Use `TestGetWorkspaceSize` with these varied combinations to exercise different code paths in the operator. "
                        "Do NOT generate multiple tests that only differ by a constant — vary dtype, shape, AND values together.",
                        "17. SAFETY CONSTRAINTS: "
                        "(a) NEVER use NaN, Inf, or -Inf as parameters to helper functions like `TensorDesc::ValueRange(low, high)` — "
                        "NaN fails the `low <= high` assertion and crashes the entire test suite. "
                        "NaN/Inf are only safe as raw tensor DATA (e.g. `float data[] = {NAN, INFINITY};`). "
                        "(b) NEVER pass nullptr to the INPUT() macro (e.g. `INPUT(nullptr)`). INPUT must always receive a valid TensorDesc. "
                        "For null-pointer validation use nullptr on the OUTPUT side only: create a TensorDesc for OUTPUT but cast it to nullptr "
                        "when passing to the function, e.g. `OP_API_UT_EXPECT(aclnnXxx, INPUT(valid_in), OUTPUT((aclTensor*)nullptr), ...)`. "
                        "The operator should return ACLNN_ERR_PARAM_NULLPTR when given a null output.",
                        "18. COVERAGE-DRIVEN GENERATION (for epoch >= 2): When an 'UNCOVERED CODE' section appears below targeting this layer, "
                        "your PRIMARY goal is to design tests that trigger those specific uncovered branches. "
                        "For each uncovered conditional (if/switch/ternary) at the listed line number: "
                        "(a) READ the source file via `cat` to see the exact branch condition. "
                        "(b) Analyze what dtype/shape/value/format/attribute combination makes the condition evaluate to the uncovered branch. "
                        "(c) Create a dedicated TEST_F with inputs that satisfy the uncovered path. "
                        "Name the test using the target line number for traceability (e.g. `CASE_L245_dtype_promotion_path`). "
                        "If no `UNCOVERED CODE` section appears or the current layer is not targeted, fall back to the DIVERSITY REQUIREMENT in Rule 16.",
                    ]

                    uncovered_section = ""
                    epoch = int(getattr(state, "epoch_current", 1) or 1)
                    if epoch > 1 and layer_id in uncovered_for_layer:
                        udata = uncovered_for_layer[layer_id]
                        files_info = udata.get("files", [])
                        snippets_text_parts: List[str] = []
                        for fi in files_info[:4]:
                            src = fi.get("source", "")
                            snippets = fi.get("snippets", [])
                            if not snippets:
                                continue
                            snippets_text_parts.append(f"\n### Source: {src}")
                            for s in snippets[:5]:
                                snippets_text_parts.append(f"\nLine {s.get('line')} is NOT covered:\n```cpp\n{s.get('context','')}\n```")
                        uncovered_section = (
                            f"\n## UNCOVERED CODE from previous epoch ({udata.get('total_uncovered', 0)} lines total in this layer)\n"
                            "Read the source files via `cat` using their full paths, then design TEST_F cases whose "
                            "dtype/shape/attribute/value combinations specifically TRIGGER these uncovered branches or lines. "
                            "For each uncovered conditional (if/switch/ternary), design TEST parameters that make it evaluate to "
                            "the uncovered branch. Include the target line number in the TEST_F name for traceability."
                            + "\n".join(snippets_text_parts)
                            + "\n"
                        )

                    cross_file_section = ""
                    if cross_file_notes:
                        notes_text = "\n".join(f"- {n}" for n in cross_file_notes)
                        cross_file_section = f"Notes from previous files in this epoch:\n{notes_text}\n"

                    build_plan_dict = plan.get("build_plan") or {}
                    layer_id_for_cmd = str(file_entry.get("layer_id", ""))
                    layer_cmds = build_plan_dict.get(layer_id_for_cmd, {}) if isinstance(build_plan_dict, dict) else {}
                    compile_cmd_for_layer = str(layer_cmds.get("compile", ""))
                    coverage_cmd_for_layer = str(layer_cmds.get("coverage", ""))

                    unified_section = ""
                    if compile_cmd_for_layer and str(file_entry.get("kind")) != "cmake":
                        unified_section = f"""
## UNIFIED GENERATE-COMPILE-TEST MODE — REFLECTION LOOP

You are NOT in a "write code then hand-off" stage. You are in a **continuous agent session**: 
write code → compile → read the REAL errors → reason about them → read the API source → fix → re-compile. 
All within this same conversation. Do NOT defer verification to a later stage.

### Per-block loop (repeat for EACH target block):

**Step 1. Write the block** via `exec_command` with heredoc (`cat >`) or `sed` to fill/replace block content.

**Step 2. Compile immediately** via `exec_command`:
```
exec_command(cmd="{compile_cmd_for_layer}")
```

- If compile succeeds → go to Step 3.
- If compile fails:
  a. **Read the FULL error output**. Do not discard anything. Identify the exact file, line, and error message.
  b. **Reason**: why did this fail? What assumption did I make that was wrong?
  c. **Read the source**: if the error mentions a function / macro / type you used incorrectly, 
     use `grep` or `cat` to look at its ACTUAL signature/implementation in the project. 
     DO NOT guess the API signature — verify it.
     Example: if `ScalarDesc(...)` fails, grep the codebase for `ScalarDesc::ScalarDesc` to see what arguments it actually accepts.
  d. **Fix the code** via `sed` or re-write the file with `cat >`.  Make the minimal correct change.
  e. **Re-run the compile command**. Repeat (a)-(d) until compile passes.

**Step 3. Run tests immediately** via `exec_command` to execute the built test binary:
```
exec_command(cmd="{compile_cmd_for_layer}")
```
(build.sh compiles AND runs tests by default — no extra `--run` flag needed. The test output will appear after the build completes.)

- If tests run cleanly (no abort, no segfault, even if some EXPECT_* fail) → go to Step 4.
- If you see **`Assertion failed`**, **`Aborted`**, **`SIGSEGV`**, **`core dumped`**, **`terminate called`**, or 
  the log suddenly stops mid-output — this is a **RUNTIME BUG** in the generated code:
  a. Read the exact error message. Pay attention to WHICH assertion failed and on WHAT line.
  b. Reason: what **input value/dtype/shape** did I pass that violated the API's contract?
  c. Read the actual source of the failing function via `cat`/`grep` to understand what inputs are valid.
  d. Fix the code.
  e. Re-run Step 2 (compile) + Step 3 (run).
  f. **Do NOT move forward until tests run to completion without abort.**

**Step 4. Check coverage** (at end of file, all blocks done) via `exec_command`:
```
exec_command(cmd="{coverage_cmd_for_layer}")
```
Read the coverage output. Identify uncovered lines that look reachable. For each reachable uncovered line,
read the source to understand the branch condition, then add a TEST_F that triggers it. Re-do Step 2+3.

### Reflection principles:
- **Write ONE block, verify it, then move to the next.** Never accumulate multiple unverified blocks.
- **NEVER guess an API signature.** If you are unsure whether a function accepts a particular dtype/shape/value,
  `cat` or `grep` its declaration or implementation NOW — don't find out the hard way via a compile error.
- **Runtime aborts are real bugs** even if compile passes. Treat assertion failures as a signal to RE-think the inputs,
  not just to tweak syntax.
- **If stuck after 2 failed repair rounds on the same issue**, write a minimal stub and move on. Do NOT loop forever.

"""

                    prompt = f"""You are generating Ascend C++ UT code for a single file.

Target file: `{file_entry['path']}`
Layer: `{file_entry['layer_id']}`
Kind: `{file_entry['kind']}`
Epoch: {getattr(state, 'epoch_current', 1)}/{getattr(state, 'epoch_total', 1)}
Target blocks to fill or revise: {', '.join(target_blocks)}

Rules:
1. You have ONLY `exec_command` as a tool. Use it for ALL operations: read files (`cat`, `head`), search (`grep`, `find`), write/edit files (`cat >`, `sed`, `tee`), compile, run tests, and check coverage.
   - For writing file content, use heredoc: `exec_command(cmd="cat > path/file.cpp << 'EOF'\n...content...\nEOF")`
2. DO NOT rewrite an existing, non-empty file or block from scratch. Fill only the listed target blocks. If a block already contains working test code, leave it untouched. Overwriting a file with fewer TEST_F cases than it contains is forbidden.
3. Keep every generated gtest or helper name traceable to the BLOCK_ID. Include the BLOCK_ID in the test name.
4. For `*.cpp`, use `// ==== BLOCK:... ====`.
5. For `CMakeLists.txt`, use `# ==== BLOCK:... ====`.
6. Prefer reading 1-2 relevant reference UT files before writing if the current file needs project-specific style.
7. Keep the output compile-oriented and minimal. Avoid placeholders like TODO.
8. Do not modify blocks that are not in the target set.
{chr(10).join(mode_rules)}
{chr(10).join(op_api_macro_rule)}

Mode guidance:
{chr(10).join(mode_guidance)}

Operator context:
```json
{json.dumps(slim_context, ensure_ascii=False, indent=2)[:8000]}
```

{_build_ws_sig_section(str(file_entry['layer_id']), slim_context)}

File plan:
```json
{json.dumps(file_entry, ensure_ascii=False, indent=2)}
```

Cases for this file:
```json
{json.dumps(file_cases, ensure_ascii=False, indent=2)}
```

Current block index:
```json
{build_block_index_json(project_root / str(file_entry['path']))}
```

Analysis plan:
```json
{json.dumps(analysis_plan, ensure_ascii=False, indent=2)}
```

Current operator UT references for this layer:
```json
{json.dumps(current_operator_ut_paths, ensure_ascii=False, indent=2)}
```

Existing test scenarios in baseline UT (DO NOT duplicate these):
```json
{json.dumps(existing_scenarios, ensure_ascii=False, indent=2)}
```

Shared harness paths for this layer:
```json
{json.dumps(shared_harness_paths, ensure_ascii=False, indent=2)}
```

Shared helper paths for this layer:
```json
{json.dumps(shared_helper_paths, ensure_ascii=False, indent=2)}
```

Operator implementation files for this layer:
```json
{json.dumps(operator_impl_paths, ensure_ascii=False, indent=2)}
```

Similar example UT paths:
```json
{json.dumps(similar_example_paths, ensure_ascii=False, indent=2)}
```

Priority reference paths:
```json
{json.dumps(reference_paths, ensure_ascii=False, indent=2)}
```

Coverage mode:
`{coverage_mode}`

Skill packet:
```text
{provider.get_stage_packet('generate_code', str(file_entry['layer_id']), generation_mode=generation_mode)}
```

{_get_few_shot_examples(str(file_entry['layer_id']))}

{infra_context_snippet}{unified_section}{uncovered_section}{case_inventory_context}{prev_error_context}{cross_file_section}
Complete the file now."""

                    stage_result = self._run_llm_session(state, prompt, project_root)
                    session_failed = False
                    if not stage_result.success:
                        session_failed = True
                        error_msg = stage_result.error or ""
                        cross_file_notes.append(
                            f"{file_id}: LLM session ended early ({error_msg[:100]}); continuing with verification"
                        )

                    if str(file_entry.get("kind")) != "cmake":
                        code_violations = self._validate_generated_code(project_root, file_entry)
                        if code_violations:
                            print(f"  ⚠️  Code safety violations in {file_id}: {'; '.join(code_violations)}")
                            violated_path = project_root / file_entry["path"]
                            if violated_path.exists():
                                quarantine = violated_path.with_suffix(violated_path.suffix + ".safety_violation.bak")
                                try:
                                    shutil.move(str(violated_path), str(quarantine))
                                except Exception:
                                    pass
                            cross_file_notes.append(
                                f"{file_id}: safety violation - {'; '.join(code_violations)[:200]}; skipping"
                            )
                            manifest.append({
                                "file_id": file_id,
                                "path": file_entry["path"],
                                "action": "error",
                                "reason": f"safety_violations: {'; '.join(code_violations)[:300]}",
                            })
                            continue

                    self._ensure_cmake_attest_registration(project_root, file_entry)

                    if str(file_entry.get("kind")) != "cmake":
                        layer_id = str(file_entry["layer_id"])
                        MAX_REPAIR_ROUNDS = 1
                        verified = False
                        for repair_round in range(MAX_REPAIR_ROUNDS):
                            compile_ok, compile_output = self._run_compile_check(project_root, layer_id, plan)
                            if not compile_ok:
                                error_summary = compile_output[:4000]
                                print(f"  ⚠️  Compile check failed for {file_id} (round {repair_round + 1}); "
                                      f"attempting LLM repair with exec_command access...")
                                repair_result = self._run_repair_with_verification(
                                    state, project_root, file_entry,
                                    error_kind="compile_error",
                                    error_text=error_summary,
                                    target_blocks=target_blocks,
                                    plan=plan,
                                )
                                if not repair_result.success:
                                    cross_file_notes.append(
                                        f"{file_id}: compile failed after repair round {repair_round + 1}"
                                    )
                                    break
                                continue

                            run_ok, run_output = self._run_binary_check(project_root, layer_id, plan)
                            abort_keywords = ("Assertion", "Aborted", "SIGSEGV", "core dumped",
                                              "terminate called", "Subprocess aborted")
                            if not run_ok or any(kw in run_output for kw in abort_keywords):
                                abort_snippet = run_output[:4000]
                                print(f"  ⚠️  Runtime abort/assertion failure for {file_id} (round {repair_round + 1}); "
                                      f"attempting in-session reflection fix...")
                                repair_result = self._run_repair_with_verification(
                                    state, project_root, file_entry,
                                    error_kind="runtime_abort",
                                    error_text=abort_snippet,
                                    target_blocks=target_blocks,
                                    plan=plan,
                                )
                                if not repair_result.success:
                                    cross_file_notes.append(
                                        f"{file_id}: runtime abort after repair round {repair_round + 1}"
                                    )
                                    break
                                continue

                            verified = True
                            break
                        else:
                            if not verified:
                                cross_file_notes.append(
                                    f"{file_id}: still failing after {MAX_REPAIR_ROUNDS} repair rounds; continuing"
                                )
                    else:
                        verified = True

                    if stage_result.success and stage_result.message:
                        cross_file_notes.append(f"{file_id}: completed successfully ({len(target_blocks)} blocks)")

                    manifest.append(
                        {
                            "file_id": file_id,
                            "path": file_entry["path"],
                            "action": "updated" if verified else "updated_unverified",
                            "target_blocks": target_blocks,
                        }
                    )

        manifest_text = _render_json({"files": manifest, "project_root": str(project_root)})
        state.save_artifact("generation_manifest.json", manifest_text)
        return StageResult(
            True,
            {"generation_manifest.json": manifest_text},
            message=f"Updated {len([item for item in manifest if item['action'] == 'updated'])} planned files",
        )


class AscendExecutionStage(AscendBaseStage):
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="execute_tests",
            display_name="Execute Tests",
            description="Run build.sh commands for enabled Ascend layers",
            prompt_template="",
            input_artifacts=["test_plan.json"],
            output_artifacts=[
                "execution_log.txt",
                "baseline_execution_log.txt",
                "generated_execution_log.txt",
                "exit_code.txt",
                "coverage_summary.json",
            ],
            tools=["exec_command"],
            allow_skip=False,
        )

    def _use_coverage(self, state, plan: Dict[str, Any]) -> bool:
        # Always collect coverage so the closed-loop optimization (analyze_results
        # emitting uncovered_code.json consumed by the next epoch's generate_code)
        # has feedback every epoch, not only on the final one. This is the key
        # driver of coverage improvement across epochs.
        return True

    def _recover_ops_info_if_missing(
        self,
        project_root: Path,
        ctx,
        label: str,
        logs: List[str],
        build_dir: str = "build",
    ) -> None:
        cov_root = f"{build_dir}/tests/ut/cov_report/cpp_utest"
        ops_info_path = f"{cov_root}/ops.info"
        check_cmd = f"test -f {shlex.quote(ops_info_path)} && echo EXISTS || echo MISSING"
        check_result = self.tool_runner.execute(
            "exec_command", {"cmd": check_cmd}, ctx, source="framework"
        )
        check_output = (check_result.output or "").strip()
        if "EXISTS" in check_output:
            return
        gcda_check = f"find {shlex.quote(build_dir)} -name '*.gcda' 2>/dev/null | head -1"
        gcda_result = self.tool_runner.execute(
            "exec_command", {"cmd": gcda_check}, ctx, source="framework"
        )
        if not (gcda_result.output or "").strip():
            logs.append(f"=== {label} no .gcda files found, skipping coverage fallback ===")
            return
        fallback_cmd = (
            f"mkdir -p {shlex.quote(cov_root)} && "
            f"rm -f {shlex.quote(ops_info_path)} && "
            f"timeout 300 lcov -c -d {shlex.quote(build_dir)} -o {shlex.quote(ops_info_path)} "
            f"--rc lcov_branch_coverage=1 2>&1 || true"
        )
        fallback_result = self.tool_runner.execute(
            "exec_command", {"cmd": fallback_cmd}, ctx, source="framework"
        )
        fallback_output = fallback_result.output if fallback_result.output else fallback_result.error or ""
        logs.append(f"=== {label} coverage recovery (direct lcov) ===\n{fallback_output[-2000:]}")

    def _default_layer_summary(
        self,
        value: float = 0.0,
        threshold: float = 80.0,
        function_coverage: Optional[float] = None,
        branch_coverage: Optional[float] = None,
        coverage_valid: bool = True,
        error_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "line_coverage": value,
            "function_coverage": function_coverage,
            "branch_coverage": branch_coverage,
            "meets_threshold": value >= threshold,
            "coverage_valid": coverage_valid,
            "coverage_error_reason": error_reason,
        }

    def _extract_summary_json(self, log_text: str) -> Dict[str, Any]:
        matches = re.findall(r"COVERAGE_SUMMARY\s+({.*})", log_text)
        for raw in reversed(matches):
            summary = _json_load(raw, {})
            if summary:
                return summary
        return {}

    def _extract_layer_coverage_from_text(
        self,
        log_text: str,
        layer: str,
        *,
        allow_generic: bool = True,
    ) -> Optional[float]:
        summary = self._extract_summary_json(log_text)
        if isinstance(summary.get("per_layer"), dict):
            layer_meta = summary["per_layer"].get(layer)
            if isinstance(layer_meta, dict) and layer_meta.get("line_coverage") is not None:
                try:
                    return float(layer_meta["line_coverage"])
                except Exception:
                    pass

        explicit_pattern = re.compile(r"COVERAGE_LAYER\s+([A-Za-z0-9_]+)\s+([0-9]+(?:\.[0-9]+)?)")
        for match in explicit_pattern.finditer(log_text):
            if match.group(1) == layer:
                return float(match.group(2))

        if not allow_generic:
            return None
        generic = re.findall(r"([0-9]+(?:\.[0-9]+)?)%", log_text)
        if generic:
            return float(generic[-1])
        return None

    def _extract_lcov_coverage(
        self,
        ctx: ToolContext,
        build_dir: str = "build",
    ) -> tuple[Optional[float], Optional[float], Optional[float], str]:
        cov_path = f"{build_dir}/tests/ut/cov_report/cpp_utest/ops.info"
        result = self.tool_runner.execute(
            "exec_command",
            {"cmd": f"if [ -f {shlex.quote(cov_path)} ]; then "
             f"lcov --summary {shlex.quote(cov_path)}; fi"},
            ctx,
        )
        output = result.output if result.output else result.error or ""
        line_match = re.search(r"lines\.+:\s*([0-9]+(?:\.[0-9]+)?)%", output, flags=re.IGNORECASE)
        func_match = re.search(r"functions\.+:\s*([0-9]+(?:\.[0-9]+)?)%", output, flags=re.IGNORECASE)
        branch_match = re.search(r"branches\.+:\s*([0-9]+(?:\.[0-9]+)?)%", output, flags=re.IGNORECASE)
        if line_match:
            return (
                float(line_match.group(1)),
                float(func_match.group(1)) if func_match else None,
                float(branch_match.group(1)) if branch_match else None,
                output,
            )
        return None, None, None, output

    def _extract_operator_lcov_coverage(
        self,
        ctx: ToolContext,
        plan: Dict[str, Any],
        layer: str,
        build_dir: str = "build",
        infra_overrides: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[float], Optional[float], Optional[float], str, bool, Optional[str]]:
        include_patterns = _operator_source_patterns(plan, layer)
        if not include_patterns:
            return None, None, None, "", False, "operator source pattern is unavailable"

        remove_patterns = _operator_remove_patterns(plan, layer, infra_overrides)
        cov_root = f"{build_dir}/tests/ut/cov_report/cpp_utest"
        source_info = f"{cov_root}/ops.info"
        extract_info = f"{cov_root}/attest_{layer}_extract.info"
        filtered_info = f"{cov_root}/attest_{layer}_filtered.info"
        include_args = " ".join(shlex.quote(pattern) for pattern in include_patterns)
        remove_args = " ".join(shlex.quote(pattern) for pattern in remove_patterns)

        cmd_parts = [
            f"if [ ! -f {shlex.quote(source_info)} ]; then",
            f"echo 'LCOV_SOURCE_MISSING {shlex.quote(source_info)}';",
            "else",
            f"rm -f {shlex.quote(extract_info)} {shlex.quote(filtered_info)};",
            f"if lcov --extract {shlex.quote(source_info)} {include_args} -o {shlex.quote(extract_info)} >/dev/null 2>&1; then",
        ]
        if remove_patterns:
            cmd_parts.append(
                f"if lcov --remove {shlex.quote(extract_info)} {remove_args} -o {shlex.quote(filtered_info)} >/dev/null 2>&1; then"
            )
            cmd_parts.append(f"lcov --summary {shlex.quote(filtered_info)} || true;")
            cmd_parts.append("else")
            cmd_parts.append(f"echo 'LCOV_REMOVE_NO_MATCH layer={shlex.quote(layer)}';")
            cmd_parts.append(f"lcov --summary {shlex.quote(extract_info)} || true;")
            cmd_parts.append("fi;")
        else:
            cmd_parts.append(f"lcov --summary {shlex.quote(extract_info)} || true;")
        cmd_parts.append("else")
        cmd_parts.append(f"echo 'LCOV_EXTRACT_NO_MATCH layer={shlex.quote(layer)}';")
        cmd_parts.append("fi;")
        cmd_parts.append("fi")

        result = self.tool_runner.execute("exec_command", {"cmd": " ".join(cmd_parts)}, ctx, source="framework")
        output = result.output if result.output else result.error or ""
        error_reason = _coverage_error_reason(output)
        line_match = re.search(r"lines\.+:\s*([0-9]+(?:\.[0-9]+)?)%", output, flags=re.IGNORECASE)
        func_match = re.search(r"functions\.+:\s*([0-9]+(?:\.[0-9]+)?)%", output, flags=re.IGNORECASE)
        branch_match = re.search(r"branches\.+:\s*([0-9]+(?:\.[0-9]+)?)%", output, flags=re.IGNORECASE)
        if line_match:
            return (
                float(line_match.group(1)),
                float(func_match.group(1)) if func_match else None,
                float(branch_match.group(1)) if branch_match else None,
                output,
                True,
                None,
            )
        return None, None, None, output, False, error_reason or "operator lcov summary has no line coverage"

    def _extract_opbase_shared_coverage(
        self,
        ctx: ToolContext,
        build_dir: str = "build",
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Extract coverage from opbase shared utility functions (infershape_*_util.*).
        Used when an operator's op_host code only does macro registration (e.g., div/mod/floor_div),
        and the actual infershape implementation is shared via opbase.
        """
        opbase_dir = f"{build_dir}/math/abs/CMakeFiles/opbase_infer_objs.dir"
        cov_dir = f"{build_dir}/tests/ut/cov_report/cpp_utest"
        result = self.tool_runner.execute(
            "exec_command",
            {"cmd": f"if [ -d {shlex.quote(opbase_dir)} ]; then find {shlex.quote(opbase_dir)} -name '*.gcda'; fi"},
            ctx,
            source="framework",
        )
        if not result or not result.output or "gcda" not in result.output:
            return None, None

        tmp_info = f"{cov_dir}/attest_opbase_shared.info"
        cmd = (
            f"mkdir -p {shlex.quote(cov_dir)} && "
            f"rm -f {shlex.quote(tmp_info)} && "
            f"lcov -c -d {shlex.quote(opbase_dir)} -o {shlex.quote(tmp_info)} --rc lcov_branch_coverage=1 "
            f">/dev/null 2>&1 && "
            f"lcov --summary {shlex.quote(tmp_info)} --rc lcov_branch_coverage=1 || true"
        )
        result = self.tool_runner.execute("exec_command", {"cmd": cmd}, ctx, source="framework")
        output = (result.output if result.output else result.error or "") if result else ""
        line_match = re.search(r"lines\.+:\s*([0-9]+(?:\.[0-9]+)?)%", output)
        func_match = re.search(r"functions\.+:\s*([0-9]+(?:\.[0-9]+)?)%", output)
        if line_match:
            return float(line_match.group(1)), float(func_match.group(1)) if func_match else None
        return None, None

    def _extract_uncovered_lines(
        self,
        ctx: ToolContext,
        plan: Dict[str, Any],
        layer: str,
        build_dir: str = "build",
        max_lines_per_file: int = 40,
        infra_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        include_patterns = _operator_source_patterns(plan, layer)
        if not include_patterns:
            return []
        cov_root = f"{build_dir}/tests/ut/cov_report/cpp_utest"
        source_info = f"{cov_root}/ops.info"
        extract_info = f"{cov_root}/attest_{layer}_extract.info"
        filtered_info = f"{cov_root}/attest_{layer}_filtered.info"
        target_info = filtered_info if _operator_remove_patterns(plan, layer, infra_overrides) else extract_info

        cmd = (
            f"if [ -f {shlex.quote(target_info)} ]; then "
            f"grep -E '^SF:|^DA:[0-9]+,0($|,)' {shlex.quote(target_info)}; fi"
        )
        result = self.tool_runner.execute("exec_command", {"cmd": cmd}, ctx, source="framework")
        output = (result.output if result.output else result.error or "") if result else ""

        files: Dict[str, Dict[str, Any]] = {}
        current_file = ""
        for raw in output.splitlines():
            line = raw.strip()
            if line.startswith("SF:"):
                current_file = line[3:]
                files.setdefault(current_file, {"path": current_file, "uncovered_lines": []})
            elif line.startswith("DA:") and current_file:
                parts = line[3:].split(",")
                if len(parts) >= 2:
                    try:
                        lineno = int(parts[0])
                        hit = int(parts[1])
                        if hit == 0:
                            if len(files[current_file]["uncovered_lines"]) < max_lines_per_file:
                                files[current_file]["uncovered_lines"].append(lineno)
                            files[current_file]["total_uncovered"] = files[current_file].get("total_uncovered", 0) + 1
                    except (ValueError, IndexError):
                        continue
        ordered = list(files.values())
        ordered.sort(key=lambda e: e.get("total_uncovered", 0), reverse=True)
        return ordered

    def _extract_uncovered_functions(
        self,
        ctx: ToolContext,
        plan: Dict[str, Any],
        layer: str,
        build_dir: str = "build",
        infra_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        include_patterns = _operator_source_patterns(plan, layer)
        if not include_patterns:
            return []
        remove_patterns = _operator_remove_patterns(plan, layer, infra_overrides)
        cov_root = f"{build_dir}/tests/ut/cov_report/cpp_utest"
        source_info = f"{cov_root}/ops.info"
        extract_info = f"{cov_root}/attest_{layer}_extract.info"
        filtered_info = f"{cov_root}/attest_{layer}_filtered.info"
        target_info = filtered_info if remove_patterns else extract_info
        include_args = " ".join(shlex.quote(p) for p in include_patterns)
        remove_args = " ".join(shlex.quote(p) for p in remove_patterns)

        if remove_patterns:
            cmd = (
                f"if [ -f {shlex.quote(target_info)} ]; then "
                f"grep -E '^SF:|^FNDA:' {shlex.quote(target_info)}; fi"
            )
        else:
            cmd = (
                f"if [ -f {shlex.quote(target_info)} ]; then "
                f"grep -E '^SF:|^FNDA:' {shlex.quote(target_info)}; fi"
            )

        result = self.tool_runner.execute("exec_command", {"cmd": cmd}, ctx, source="framework")
        output = result.output if result.output else result.error or ""
        if not output.strip():
            if remove_patterns:
                fallback = extract_info
            else:
                fallback = target_info
            cmd = (
                f"if [ -f {shlex.quote(fallback)} ]; then "
                f"grep -E '^SF:|^FNDA:' {shlex.quote(fallback)}; fi"
            )
            result = self.tool_runner.execute("exec_command", {"cmd": cmd}, ctx, source="framework")
            output = result.output if result.output else result.error or ""

        current_file = ""
        uncovered = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("SF:"):
                current_file = line[3:]
            elif line.startswith("FNDA:"):
                parts = line[5:].split(",")
                if len(parts) == 2:
                    try:
                        hit_count = int(parts[0])
                        func_name = parts[1]
                        if hit_count == 0:
                            short_file = current_file.split("/")[-1] if current_file else "?"
                            uncovered.append(f"{short_file}:{func_name}")
                    except (ValueError, IndexError):
                        pass
        return uncovered[:20]

    def _build_run_summary(
        self,
        plan: Dict[str, Any],
        project_root: Path,
        per_layer: Dict[str, Dict[str, Any]],
        exit_code: int,
    ) -> Dict[str, Any]:
        layer_threshold = _layer_coverage_threshold(plan)
        overall_threshold = _overall_coverage_threshold(plan)
        normalized: Dict[str, Dict[str, Any]] = {}
        invalid_layers: Dict[str, str] = {}
        for layer in plan.get("enabled_layers", []):
            layer_meta = per_layer.get(str(layer)) or self._default_layer_summary(
                threshold=layer_threshold,
                coverage_valid=False,
                error_reason="coverage was not collected for this layer",
            )
            value = float(layer_meta.get("line_coverage", 0.0))
            coverage_valid = bool(layer_meta.get("coverage_valid", True))
            error_reason = layer_meta.get("coverage_error_reason")
            normalized[str(layer)] = self._default_layer_summary(
                value,
                layer_threshold,
                function_coverage=layer_meta.get("function_coverage"),
                branch_coverage=layer_meta.get("branch_coverage"),
                coverage_valid=coverage_valid,
                error_reason=error_reason,
            )
            if layer_meta.get("uncovered_lines"):
                normalized[str(layer)]["uncovered_lines"] = layer_meta["uncovered_lines"]
            if layer_meta.get("uncovered_functions"):
                normalized[str(layer)]["uncovered_functions"] = layer_meta["uncovered_functions"]
            if not coverage_valid:
                invalid_layers[str(layer)] = str(error_reason or "coverage is invalid")

        overall = 0.0
        valid_layers = [item for item in normalized.values() if item.get("coverage_valid", True)]
        if valid_layers:
            overall = sum(item["line_coverage"] for item in valid_layers) / len(valid_layers)
        coverage_valid = len(valid_layers) == len(normalized) and not invalid_layers

        return {
            "project_root": str(project_root),
            "exit_code": exit_code,
            "coverage_valid": coverage_valid,
            "coverage_errors": invalid_layers,
            "threshold": overall_threshold,
            "thresholds": {
                "overall": overall_threshold,
                "per_layer": layer_threshold,
            },
            "per_layer": normalized,
            "overall": {
                "line_coverage": overall,
                "meets_threshold": coverage_valid and overall >= overall_threshold,
                "coverage_valid": coverage_valid,
            },
        }

    def _run_for_root(
        self,
        plan: Dict[str, Any],
        project_root: Path,
        use_coverage: bool,
        label: str,
        state_obj: Any = None,
    ) -> Dict[str, Any]:
        build_plan = plan.get("build_plan") or {}
        ctx = ToolContext(cwd=str(project_root), auto_approve=True)
        logs: List[str] = []
        exit_code = 0
        per_layer: Dict[str, Dict[str, Any]] = {}

        infra_overrides: Optional[Dict[str, Any]] = None
        if state_obj is not None:
            try:
                infra_ctx = _load_json_artifact(state_obj, "infrastructure_context.json", default={})
                infra_overrides = infra_ctx.get("lcov_overrides") or None
            except Exception:
                pass

        op_name = plan.get("op_name", "")
        if op_name:
            build_dir = f"build_{label}_{op_name}"
        else:
            build_dir = "build"
        build_env_prefix = ""
        build_sh_patched = False
        build_sh_path = project_root / "build.sh"
        build_sh_backup: Optional[Path] = None
        if build_dir != "build":
            build_out_dir = f"build_out_{label}_{op_name}"
            build_env_prefix = (
                f"BUILD_PATH={shlex.quote(str(project_root / build_dir))} "
                f"BUILD_OUT_PATH={shlex.quote(str(project_root / build_out_dir))} "
            )
            logs.append(f"=== {label} using isolated build dir: {build_dir} ===")
            # Backup build.sh before patching (restore in finally block)
            if build_sh_path.exists():
                build_sh_backup = build_sh_path.with_suffix(".sh.v9.bak")
                shutil.copy2(build_sh_path, build_sh_backup)
            _patch_build_sh_for_isolation(project_root)
            build_sh_patched = build_sh_backup is not None
            if _ensure_cann_cmake_available(project_root):
                logs.append(f"=== {label} ensured local cann-cmake in third_party/ ===")
            else:
                logs.append(f"=== {label} WARNING: local cann-cmake not available, build may need git clone ===")

        # Phase 1: compile (skipped when combined coverage will rebuild anyway)
        combined = build_plan.get("_combined", {})
        combined_compile_cmd = combined.get("combined_compile", "")
        combined_cov_cmd = combined.get("combined_coverage", "")
        compile_failed = False
        if combined_compile_cmd and not (use_coverage and combined_cov_cmd):
            result = self.tool_runner.execute(
                "exec_command", {"cmd": f"{build_env_prefix}{combined_compile_cmd}"}, ctx, source="framework"
            )
            logs.append(f"=== {label} combined compile ===\n{result.output if result.output else result.error or ''}")
            if not result.ok:
                exit_code = 1
                compile_failed = True
        elif not combined_compile_cmd:
            for layer in plan.get("enabled_layers", []):
                commands = build_plan.get(layer)
                if not commands:
                    continue
                compile_cmd = commands.get("compile", "")
                if compile_cmd:
                    result = self.tool_runner.execute(
                        "exec_command", {"cmd": f"{build_env_prefix}{compile_cmd}"}, ctx, source="framework"
                    )
                    logs.append(f"=== {label} {layer} compile ===\n{result.output if result.output else result.error or ''}")
                    if not result.ok:
                        exit_code = 1
                        compile_failed = True

        # Phase 1.5: If compile failed, isolate generated attest files to prevent them from
        # breaking the overall coverage target. Move problematic *_attest.cpp files aside
        # and retry compile with --noexec to extract coverage from whatever still compiles.
        attest_files_quarantined: list = []
        if compile_failed:
            attest_dir = project_root / "math" / plan.get("op_name", "") / "tests" / "ut"
            if attest_dir.exists():
                for cpp_file in attest_dir.rglob("*_attest.cpp"):
                    backup = cpp_file.with_suffix(".cpp.bak")
                    shutil.move(str(cpp_file), str(backup))
                    attest_files_quarantined.append((cpp_file, backup))
            if attest_files_quarantined:
                logs.append(
                    f"=== {label} quarantined {len(attest_files_quarantined)} attest files "
                    f"to retry compile ==="
                )
                if combined_compile_cmd:
                    result = self.tool_runner.execute(
                        "exec_command", {"cmd": f"{build_env_prefix}{combined_compile_cmd}"}, ctx, source="framework"
                    )
                    logs.append(f"=== {label} retry compile (quarantined) ===\n{result.output if result.output else result.error or ''}")
                    if result.ok:
                        exit_code = 0
                        compile_failed = False

        coverage_error = None
        try:
            # Phase 2: combined coverage build (single invocation preserves all layers)
            if use_coverage and combined_cov_cmd:
                clean_gcda = f"find {shlex.quote(build_dir)} -name '*.gcda' -delete 2>/dev/null; true"
                self.tool_runner.execute("exec_command", {"cmd": clean_gcda}, ctx, source="framework")
                if not compile_failed:
                    cov_result = self.tool_runner.execute(
                        "exec_command", {"cmd": f"{build_env_prefix}{combined_cov_cmd}"}, ctx, source="framework"
                    )
                    cov_output = cov_result.output if cov_result.output else cov_result.error or ""
                    logs.append(f"=== {label} combined coverage ===\n{cov_output}")
                    _ce = _coverage_error_reason(cov_output)
                    if coverage_error != "ZeroTestsRegistered":
                        coverage_error = _ce
                    cov_ok = bool(cov_result.ok)
                    if not cov_ok:
                        if not coverage_error:
                            exit_code = 1
                            coverage_error = "coverage command failed"
                        self._recover_ops_info_if_missing(
                            project_root, ctx, label, logs, build_dir=build_dir
                        )
                    if re.search(r"Running\s+0\s+tests\s+from\s+0\s+test\s+suite", cov_output):
                        logs.append(
                            f"=== {label} ZERO TESTS DETECTED: test binary has 0 TEST_F registrations. "
                            "This likely means: (a) attest-generated *_attest.cpp files were quarantined "
                            "due to compile errors, or (b) CMakeLists.txt does not pick up attest test "
                            "files. Check compile errors in build log above. ==="
                        )
                        cov_ok = False
                        coverage_error = "ZeroTestsRegistered"
                else:
                    cov_result = None
                    cov_output = ""
                    coverage_error = "compile failed, coverage skipped"
                    cov_ok = False
            else:
                cov_result = None
                cov_output = ""
                coverage_error = None
                cov_ok = True

            # Phase 3: per-layer lcov extraction from the combined ops.info
            for layer in plan.get("enabled_layers", []):
                commands = build_plan.get(layer)
                if not commands:
                    continue

                if use_coverage and (combined_cov_cmd or commands.get("coverage")):
                    if not combined_cov_cmd:
                        # Fallback: per-layer coverage (no combined command available)
                        clean_gcda = f"find {shlex.quote(build_dir)} -name '*.gcda' -delete 2>/dev/null; true"
                        self.tool_runner.execute("exec_command", {"cmd": clean_gcda}, ctx, source="framework")
                        cov_result_local = self.tool_runner.execute(
                            "exec_command", {"cmd": f"{build_env_prefix}{commands['coverage']}"}, ctx, source="framework"
                        )
                        cov_output_local = cov_result_local.output if cov_result_local.output else cov_result_local.error or ""
                        logs.append(f"=== {label} {layer} coverage ===\n{cov_output_local}")
                        _ce = _coverage_error_reason(cov_output_local)
                        if coverage_error != "ZeroTestsRegistered":
                            coverage_error = _ce
                        cov_ok = bool(cov_result_local.ok)
                        if not cov_ok:
                            if not coverage_error:
                                exit_code = 1
                                coverage_error = "coverage command failed"
                            self._recover_ops_info_if_missing(
                                project_root, ctx, label, logs, build_dir=build_dir
                            )
                    else:
                        cov_result_local = cov_result
                        _ce = _coverage_error_reason(cov_output)
                        if coverage_error != "ZeroTestsRegistered":
                            coverage_error = _ce

                    layer_value = None
                    function_value = None
                    branch_value = None
                    layer_source = ""
                    operator_lcov_valid = False
                    operator_lcov_error = None
                    (
                        layer_value,
                        function_value,
                        branch_value,
                        operator_lcov_output,
                        operator_lcov_valid,
                        operator_lcov_error,
                    ) = self._extract_operator_lcov_coverage(ctx, plan, str(layer), build_dir=build_dir, infra_overrides=infra_overrides)
                    
                    
                    if layer_value is not None:
                        layer_source = "operator_lcov"
                    if operator_lcov_output.strip():
                        logs.append(f"=== {label} {layer} operator lcov summary ===\n{operator_lcov_output}")

                    if layer_value is None:
                        layer_value = self._extract_layer_coverage_from_text(cov_output, str(layer), allow_generic=False)
                        if layer_value is not None:
                            layer_source = "explicit_log"
                    lcov_output = ""
                    if layer_value is None and not operator_lcov_error and not coverage_error:
                        layer_value, function_value, branch_value, lcov_output = self._extract_lcov_coverage(ctx, build_dir=build_dir)
                        if layer_value is not None:
                            layer_source = "global_lcov"
                        if lcov_output.strip():
                            logs.append(f"=== {label} {layer} lcov summary ===\n{lcov_output}")
                        coverage_error = _coverage_error_reason(lcov_output)
                    if layer_value is not None:
                        layer_valid = (cov_ok or operator_lcov_valid) and (
                            layer_source == "operator_lcov"
                            or layer_source == "explicit_log"
                            or (layer_source == "global_lcov" and not coverage_error)
                        )
                        per_layer[str(layer)] = self._default_layer_summary(
                            layer_value,
                            function_coverage=function_value,
                            branch_coverage=branch_value,
                            coverage_valid=layer_valid,
                            error_reason=None if layer_valid else coverage_error or operator_lcov_error,
                        )
                    else:
                        per_layer[str(layer)] = self._default_layer_summary(
                            threshold=_layer_coverage_threshold(plan),
                            coverage_valid=False,
                            error_reason=coverage_error or operator_lcov_error or "coverage was not found for this layer",
                        )

                    uncovered_funcs = self._extract_uncovered_functions(ctx, plan, str(layer), build_dir=build_dir, infra_overrides=infra_overrides)
                    if uncovered_funcs:
                        per_layer[str(layer)]["uncovered_functions"] = uncovered_funcs
                    uncovered_lines = self._extract_uncovered_lines(ctx, plan, str(layer), build_dir=build_dir, infra_overrides=infra_overrides)
                    if uncovered_lines:
                        per_layer[str(layer)]["uncovered_lines"] = uncovered_lines
                        total_uncovered = sum(e.get("total_uncovered", 0) for e in uncovered_lines)
                        logs.append(
                            f"=== {label} {layer} uncovered_lines: {total_uncovered} lines across {len(uncovered_lines)} file(s) ==="
                        )
            # Phase 2.55: Preserve lcov info files for analyze_results (data-driven next epoch)
            lcov_info_artifacts: Dict[str, str] = {}
            if state_obj is not None and getattr(state_obj, "artifacts_dir", None):
                for layer in plan.get("enabled_layers", []):
                    cov_root = Path(project_root) / build_dir / "tests" / "ut" / "cov_report" / "cpp_utest"
                    for suffix in ("filtered", "extract"):
                        info_file = cov_root / f"attest_{layer}_{suffix}.info"
                        if info_file.exists():
                            epoch = max(1, int(getattr(state_obj, "epoch_current", 1) or 1))
                            rel_artifact = f"generated_{layer}_{suffix}.ep{epoch}.lcov.info"
                            try:
                                shutil.copy2(str(info_file), str(state_obj.artifacts_dir / rel_artifact))
                                lcov_info_artifacts[f"{layer}_{suffix}"] = {
                                    "artifact": rel_artifact,
                                    "absolute_path": str(state_obj.artifacts_dir / rel_artifact),
                                }
                            except Exception as e:
                                logs.append(f"=== {label} WARNING: failed to preserve {info_file}: {e} ===")
            if lcov_info_artifacts:
                logs.append(f"=== {label} preserved lcov info artifacts: {list(lcov_info_artifacts.keys())} ===")
        finally:
            # Phase 2.5: Restore quarantined attest files
            for cpp_file, backup in attest_files_quarantined:
                if backup.exists():
                    if cpp_file.exists():
                        cpp_file.unlink()
                    shutil.move(str(backup), str(cpp_file))
            if attest_files_quarantined:
                logs.append(
                    f"=== {label} restored {len(attest_files_quarantined)} quarantined attest files ==="
                )
            # Phase 2.6: Restore original build.sh if it was patched
            if build_sh_patched and build_sh_backup and build_sh_backup.exists():
                try:
                    shutil.copy2(build_sh_backup, build_sh_path)
                    build_sh_backup.unlink()
                    logs.append(f"=== {label} restored original build.sh ===")
                except Exception as e:
                    logs.append(f"=== {label} WARNING: failed to restore build.sh: {e} ===")
            if build_dir != "build":
                build_out_dir = f"build_out_{label}_{op_name}"
                cleanup_cmd = f"rm -rf {shlex.quote(build_dir)} {shlex.quote(build_out_dir)}"
                self.tool_runner.execute("exec_command", {"cmd": cleanup_cmd}, ctx, source="framework")
                logs.append(f"=== {label} cleaned up isolated build dir: {build_dir} ===")

        log_text = "\n\n".join(logs)
        operator_lcov_recovered = bool(per_layer) and all(
            layer_meta.get("coverage_valid", False) and layer_meta.get("line_coverage", 0) > 0
            for layer_meta in per_layer.values()
        )
        if exit_code != 0 and operator_lcov_recovered:
            exit_code = 0
        return {
            "log_text": log_text,
            "exit_code": exit_code,
            "summary": self._build_run_summary(plan, project_root, per_layer, exit_code),
        }

    def _build_compare_summary(
        self,
        plan: Dict[str, Any],
        baseline: Dict[str, Any],
        generated: Dict[str, Any],
    ) -> Dict[str, Any]:
        layer_threshold = _layer_coverage_threshold(plan)
        overall_threshold = _overall_coverage_threshold(plan)
        per_layer: Dict[str, Dict[str, Any]] = {}
        baseline_values: List[float] = []
        generated_values: List[float] = []
        generated_meets_layer_threshold = True
        generated_ge_baseline = True
        invalid_layers: Dict[str, Dict[str, str]] = {}
        skipped_layers: Dict[str, str] = {}

        for layer in plan.get("enabled_layers", []):
            baseline_meta = dict(
                (baseline.get("summary") or {}).get("per_layer", {}).get(layer)
                or self._default_layer_summary(threshold=layer_threshold)
            )
            generated_meta = dict(
                (generated.get("summary") or {}).get("per_layer", {}).get(layer)
                or self._default_layer_summary(threshold=layer_threshold)
            )
            baseline_value = float(baseline_meta.get("line_coverage", 0.0))
            generated_value = float(generated_meta.get("line_coverage", 0.0))
            baseline_func = baseline_meta.get("function_coverage")
            generated_func = generated_meta.get("function_coverage")
            baseline_branch = baseline_meta.get("branch_coverage")
            generated_branch = generated_meta.get("branch_coverage")
            baseline_valid = bool(baseline_meta.get("coverage_valid", True))
            generated_valid = bool(generated_meta.get("coverage_valid", True))
            baseline_uncovered = not baseline_valid and baseline_value == 0.0
            if baseline_valid and generated_valid:
                baseline_values.append(baseline_value)
                generated_values.append(generated_value)
            elif baseline_uncovered:
                skipped_layers[str(layer)] = "baseline has no instrumented code for this layer"
                if generated_valid:
                    baseline_values.append(baseline_value)
                    generated_values.append(generated_value)
            else:
                invalid_layers[str(layer)] = {
                    "baseline": str(baseline_meta.get("coverage_error_reason") or ""),
                    "generated": str(generated_meta.get("coverage_error_reason") or ""),
                }
            if not baseline_uncovered:
                generated_meets_layer_threshold = (
                    generated_meets_layer_threshold and generated_valid and generated_value >= layer_threshold
                )
                generated_ge_baseline = (
                    generated_ge_baseline and baseline_valid and generated_valid and generated_value >= baseline_value
                )
            per_layer[str(layer)] = {
                "baseline": self._default_layer_summary(
                    baseline_value,
                    layer_threshold,
                    baseline_func,
                    baseline_branch,
                    baseline_valid,
                    baseline_meta.get("coverage_error_reason"),
                ),
                "generated": self._default_layer_summary(
                    generated_value,
                    layer_threshold,
                    generated_func,
                    generated_branch,
                    generated_valid,
                    generated_meta.get("coverage_error_reason"),
                ),
                "delta_line_coverage": round(generated_value - baseline_value, 4),
                "delta_function_coverage": (
                    round(float(generated_func) - float(baseline_func), 4)
                    if baseline_func is not None and generated_func is not None
                    else None
                ),
                "delta_branch_coverage": (
                    round(float(generated_branch) - float(baseline_branch), 4)
                    if baseline_branch is not None and generated_branch is not None
                    else None
                ),
            }
            if baseline_meta.get("uncovered_lines"):
                per_layer[str(layer)]["baseline_uncovered_lines"] = baseline_meta["uncovered_lines"]
            if generated_meta.get("uncovered_lines"):
                per_layer[str(layer)]["generated_uncovered_lines"] = generated_meta["uncovered_lines"]
            if baseline_meta.get("uncovered_functions"):
                per_layer[str(layer)]["baseline_uncovered_functions"] = baseline_meta["uncovered_functions"]
            if generated_meta.get("uncovered_functions"):
                per_layer[str(layer)]["generated_uncovered_functions"] = generated_meta["uncovered_functions"]

        baseline_avg = sum(baseline_values) / len(baseline_values) if baseline_values else 0.0
        generated_avg = sum(generated_values) / len(generated_values) if generated_values else 0.0
        coverage_valid = not invalid_layers and (bool(baseline_values) or bool(generated_values))
        generated_meets_overall_threshold = coverage_valid and generated_avg >= overall_threshold
        generated_meets_threshold = generated_meets_layer_threshold and generated_meets_overall_threshold

        return {
            "mode": "before_after_compare",
            "coverage_valid": coverage_valid,
            "coverage_errors": invalid_layers,
            "skipped_layers": skipped_layers,
            "threshold": overall_threshold,
            "thresholds": {
                "overall": overall_threshold,
                "per_layer": layer_threshold,
            },
            "compare_scope": plan.get("compare_scope", []),
            "baseline_root": baseline.get("summary", {}).get("project_root", ""),
            "generated_root": generated.get("summary", {}).get("project_root", ""),
            "runs": {
                "baseline": baseline.get("summary", {}),
                "generated": generated.get("summary", {}),
            },
            "per_layer": per_layer,
            "overall": {
                "baseline_avg": baseline_avg,
                "generated_avg": generated_avg,
                "delta_line_coverage": round(generated_avg - baseline_avg, 4),
                "generated_meets_threshold": generated_meets_threshold,
                "generated_meets_overall_threshold": generated_meets_overall_threshold,
                "generated_meets_layer_threshold": generated_meets_layer_threshold,
                "generated_ge_baseline": generated_ge_baseline,
                "coverage_valid": coverage_valid,
            },
        }

    def execute(self, state) -> StageResult:
        plan = _load_json_artifact(state, "test_plan.json")
        if not (plan.get("build_plan") or {}):
            return StageResult(False, {}, error="Missing build_plan in test_plan.json")

        use_coverage = self._use_coverage(state, plan)
        compare_mode = _coverage_mode(plan) == "before_after_compare"

        if compare_mode:
            baseline_root = Path((plan.get("baseline") or {}).get("project_root") or state.project_root)
            generated_root = Path((plan.get("generated") or {}).get("project_root") or _generated_repo_root(state))

            baseline_run = self._run_for_root(plan, baseline_root, use_coverage, "baseline", state)
            generated_run = self._run_for_root(plan, generated_root, use_coverage, "generated", state)

            combined_log = "\n\n".join(
                [
                    baseline_run["log_text"],
                    generated_run["log_text"],
                ]
            ).strip()
            exit_payload = {
                "baseline": int(baseline_run["exit_code"]),
                "generated": int(generated_run["exit_code"]),
                "overall": 0 if baseline_run["exit_code"] == 0 and generated_run["exit_code"] == 0 else 1,
            }
            coverage_summary = self._build_compare_summary(plan, baseline_run, generated_run)

            state.save_artifact("baseline_execution_log.txt", baseline_run["log_text"])
            state.save_artifact("generated_execution_log.txt", generated_run["log_text"])
        else:
            generated_root = Path((plan.get("generated") or {}).get("project_root") or state.project_root)
            run_result = self._run_for_root(plan, generated_root, use_coverage, "generated", state)
            combined_log = run_result["log_text"]
            exit_payload = {"overall": int(run_result["exit_code"])}
            coverage_summary = run_result["summary"]

        coverage_text = _render_json(coverage_summary)
        state.save_artifact("execution_log.txt", combined_log)
        state.save_artifact("exit_code.txt", _render_json(exit_payload))
        state.save_artifact("coverage_summary.json", coverage_text)
        # Save useful test-result / coverage info for next-epoch context (P4).
        # In compare_mode, prefer the generated run log (has gtest results,
        # coverage summary, and uncovered lines) over the combined log so the
        # next epoch LLM gets test outcomes, not build noise.
        if compare_mode:
            src_log = generated_run["log_text"]
        else:
            src_log = combined_log
        prev_epoch_content = _extract_execution_summary(src_log, max_chars=8000)
        state.save_artifact("prev_epoch_error_log.txt", prev_epoch_content)
        outputs = {
            "execution_log.txt": combined_log,
            "exit_code.txt": _render_json(exit_payload),
            "coverage_summary.json": coverage_text,
        }
        if compare_mode:
            outputs["baseline_execution_log.txt"] = baseline_run["log_text"]
            outputs["generated_execution_log.txt"] = generated_run["log_text"]
        coverage_recovered = (
            not exit_payload.get("overall", 1) == 0
            and coverage_summary.get("coverage_valid", False)
        )
        if exit_payload.get("overall", 1) == 0:
            message = "All configured build commands passed"
        elif coverage_recovered:
            message = "Coverage recovered via lcov extraction; build.sh --cov had non-zero exit"
        else:
            message = "Build or test commands reported failures"
        return StageResult(
            True,
            outputs,
            message=message,
        )


class AscendAnalysisStage(AscendBaseStage):
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="analyze_results",
            display_name="Analyze Results",
            description="Analyze Ascend build/gtest logs and coverage summary",
            prompt_template="",
            input_artifacts=["execution_log.txt", "exit_code.txt", "coverage_summary.json", "test_plan.json"],
            output_artifacts=["analysis.md", "analysis_plan.json"],
            tools=[],
            allow_skip=False,
        )

    def _collect_uncovered_code(
        self,
        state,
        plan: Dict[str, Any],
        coverage: Dict[str, Any],
        context_window: int = 6,
    ) -> Dict[str, Any]:
        compare_mode = coverage.get("mode") == "before_after_compare"
        generated_meta_lookup = (coverage.get("runs") or {}).get("generated") or {}
        per_layer_generated = generated_meta_lookup.get("per_layer", {}) if compare_mode else coverage.get("per_layer", {})

        generated_root_str = (
            (plan.get("generated") or {}).get("project_root") or str(_generated_repo_root(state))
        )
        generated_root = Path(generated_root_str)

        result: Dict[str, Any] = {"layers": [], "total_uncovered_lines": 0, "total_files": 0}

        for layer in plan.get("enabled_layers", []):
            layer = str(layer)
            layer_data = per_layer_generated.get(layer) or {}
            uncovered_entries = (
                layer_data.get("generated_uncovered_lines")
                or layer_data.get("uncovered_lines")
                or []
            )
            if not uncovered_entries:
                continue
            layer_files: List[Dict[str, Any]] = []
            layer_total = 0
            for entry in uncovered_entries:
                if not isinstance(entry, dict):
                    continue
                source_path = entry.get("path") or ""
                uncovered_lines = entry.get("uncovered_lines") or []
                if not source_path or not uncovered_lines:
                    continue
                resolved = self._resolve_source_path(generated_root, source_path, plan)
                if not resolved or not resolved.exists():
                    layer_files.append({
                        "source": source_path,
                        "resolved": str(resolved) if resolved else None,
                        "uncovered_count": entry.get("total_uncovered", len(uncovered_lines)),
                        "uncovered_lines": uncovered_lines,
                        "snippets": [],
                    })
                    continue
                try:
                    all_lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
                except Exception:
                    all_lines = []
                snippets: List[Dict[str, Any]] = []
                for lineno in sorted(set(uncovered_lines))[:20]:
                    start = max(0, int(lineno) - context_window - 1)
                    end = min(len(all_lines), int(lineno) + context_window)
                    snippet = "\n".join(
                        f"{i+1:4d}{'>' if (i+1) == int(lineno) else ' '} {all_lines[i]}"
                        for i in range(start, end)
                    )
                    snippets.append({"line": int(lineno), "context": snippet})
                total_uncovered = entry.get("total_uncovered", len(uncovered_lines))
                layer_files.append({
                    "source": source_path,
                    "resolved": str(resolved),
                    "uncovered_count": total_uncovered,
                    "uncovered_lines": uncovered_lines,
                    "snippets": snippets,
                })
                layer_total += total_uncovered
            if layer_files:
                result["layers"].append({
                    "layer": layer,
                    "files": layer_files,
                    "total_uncovered": layer_total,
                })
                result["total_uncovered_lines"] += layer_total
                result["total_files"] += len(layer_files)
        return result

    def _resolve_source_path(
        self,
        generated_root: Path,
        source_path: str,
        plan: Dict[str, Any],
    ) -> Optional[Path]:
        normalized = source_path.replace("\\", "/")
        candidates = [
            generated_root / normalized,
            Path(normalized),
        ]
        op_name = plan.get("op_name", "")
        category = plan.get("category", "")
        for anchor in (f"{category}/{op_name}/", f"math/{op_name}/"):
            idx = normalized.find(anchor)
            if idx >= 0:
                rel = normalized[idx:]
                candidates.append(generated_root / rel)
        if "/op_host/" in normalized:
            idx = normalized.find("/op_host/")
            candidates.append(generated_root / f"math/{op_name}{normalized[idx:]}")
        if "/op_api/" in normalized:
            idx = normalized.find("/op_api/")
            candidates.append(generated_root / f"math/{op_name}{normalized[idx:]}")
        seen: set = set()
        for c in candidates:
            if c and str(c) not in seen:
                seen.add(str(c))
                if c.exists():
                    return c
        return candidates[0] if candidates else None

    def _map_failure(self, state, plan: Dict[str, Any], log_text: str) -> Dict[str, Any]:
        path_to_file_id = _path_to_file_id(plan)
        cases = {str(item["block_id"]): item for item in plan.get("cases", []) if isinstance(item, dict)}
        block_match = re.search(r"(CASE_[0-9A-Za-z_]+)", log_text)
        block_id = _normalize_case_block_id(block_match.group(1)) if block_match else "HEADER"
        case = cases.get(block_id, {})
        file_id = case.get("file_id", "")
        layer_id = case.get("layer_id", "")

        if not file_id:
            for path, candidate_file_id in path_to_file_id.items():
                if path in log_text:
                    file_id = candidate_file_id
                    break
        if not layer_id and file_id:
            for entry in plan.get("files", []):
                if entry.get("file_id") == file_id:
                    layer_id = entry.get("layer_id", "")
                    break
        if not file_id:
            file_id = next((entry.get("file_id") for entry in plan.get("files", []) if isinstance(entry, dict)), "")
        if not layer_id:
            layer_id = next((layer for layer in plan.get("enabled_layers", [])), "")

        error_type = _infer_execution_error_type(log_text)

        test_name = ""
        failed_match = re.search(r"\[\s*FAILED\s*\]\s+([^\s]+)", log_text)
        if failed_match:
            test_name = failed_match.group(1)

        return {
            "test": test_name or block_id,
            "block_id": block_id,
            "file_id": file_id,
            "layer_id": layer_id,
            "error_type": error_type,
            "action": "rewrite_block" if block_id.startswith("CASE_") else "fix_dependency",
            "note": "failure mapped from execution log",
        }

    def _generated_project_root(self, state, plan: Dict[str, Any], coverage: Dict[str, Any]) -> Path:
        if coverage.get("mode") == "before_after_compare":
            generated_root = str(coverage.get("generated_root") or (plan.get("generated") or {}).get("project_root") or "")
            if generated_root:
                return Path(generated_root)
        generated_root = str((plan.get("generated") or {}).get("project_root") or "")
        if generated_root:
            return Path(generated_root)
        return Path(state.project_root)

    def _coverage_failures(self, state, plan: Dict[str, Any], coverage: Dict[str, Any]) -> List[Dict[str, Any]]:
        failures: List[Dict[str, Any]] = []
        deferred = set(_deferred_set(plan))
        block_status: Dict[str, Dict[str, Any]] = {}
        project_root = self._generated_project_root(state, plan, coverage)
        layer_threshold = _layer_coverage_threshold(coverage if coverage else plan)
        overall_threshold = _overall_coverage_threshold(coverage if coverage else plan)
        for file_entry in plan.get("files", []):
            path = project_root / str(file_entry.get("path", ""))
            for entry in build_block_entries(path):
                block_status[str(entry["block_id"])] = entry

        context = _load_json_artifact(state, "operator_context.json", default={})
        is_aclnn_exclude = bool(context.get("is_aclnn_exclude"))

        compare_mode = coverage.get("mode") == "before_after_compare"
        for layer in plan.get("enabled_layers", []):
            layer = str(layer)
            if is_aclnn_exclude and layer == "op_host":
                continue
            meta = (coverage.get("per_layer") or {}).get(layer, {})
            if compare_mode:
                generated_meta = meta.get("generated", {}) if isinstance(meta, dict) else {}
                baseline_meta = meta.get("baseline", {}) if isinstance(meta, dict) else {}
                generated_cov = float(generated_meta.get("line_coverage", 0.0))
                baseline_cov = float(baseline_meta.get("line_coverage", 0.0))
                if generated_cov >= layer_threshold and generated_cov >= baseline_cov:
                    continue
                note = (
                    f"generated layer coverage {generated_cov}% below "
                    f"required max({layer_threshold}%, baseline {baseline_cov}%)"
                )
                uncovered = generated_meta.get("uncovered_functions") or []
                if uncovered:
                    note += f". Uncovered functions: {', '.join(uncovered[:5])}"
            else:
                generated_cov = float(meta.get("line_coverage", 0.0)) if isinstance(meta, dict) else 0.0
                if generated_cov >= layer_threshold:
                    continue
                note = f"line coverage {generated_cov}% below threshold"
                uncovered = meta.get("uncovered_functions") or []
                if uncovered:
                    note += f". Uncovered functions: {', '.join(uncovered[:5])}"

            selected_case: Optional[Dict[str, Any]] = None
            fallback_case: Optional[Dict[str, Any]] = None
            for case in plan.get("cases", []):
                if not isinstance(case, dict):
                    continue
                if case.get("layer_id") != layer:
                    continue
                fallback_case = fallback_case or case
                block_id = str(case.get("block_id", ""))
                if block_id not in deferred:
                    continue
                entry = block_status.get(block_id)
                if entry and entry.get("status") == "placeholder":
                    selected_case = case
                    break

            selected_case = selected_case or fallback_case
            if selected_case:
                block_id = str(selected_case.get("block_id", "HEADER"))
                failures.append(
                    {
                        "test": selected_case.get("test_name_hint", block_id),
                        "block_id": block_id,
                        "file_id": selected_case.get("file_id", ""),
                        "layer_id": layer,
                        "error_type": "CoverageGap",
                        "action": "add_case",
                        "note": note,
                    }
                )

        if compare_mode and not failures:
            generated_avg = float((coverage.get("overall") or {}).get("generated_avg", 0.0))
            if generated_avg < overall_threshold:
                weakest_layer = ""
                weakest_value = float("inf")
                for layer in plan.get("enabled_layers", []):
                    layer_meta = (coverage.get("per_layer") or {}).get(str(layer), {})
                    generated_meta = layer_meta.get("generated", {}) if isinstance(layer_meta, dict) else {}
                    value = float(generated_meta.get("line_coverage", 0.0))
                    if value < weakest_value:
                        weakest_value = value
                        weakest_layer = str(layer)

                selected_case = next(
                    (
                        case for case in plan.get("cases", [])
                        if isinstance(case, dict) and case.get("layer_id") == weakest_layer
                    ),
                    None,
                )
                if selected_case:
                    block_id = str(selected_case.get("block_id", "HEADER"))
                    failures.append(
                        {
                            "test": selected_case.get("test_name_hint", block_id),
                            "block_id": block_id,
                            "file_id": selected_case.get("file_id", ""),
                            "layer_id": weakest_layer,
                            "error_type": "CoverageGap",
                            "action": "add_case",
                            "note": (
                                f"overall generated coverage {generated_avg}% below "
                                f"threshold {overall_threshold}%"
                            ),
                        }
                    )
        return failures[:6]

    def _coverage_progress_metrics(self, plan: Dict[str, Any], coverage: Dict[str, Any]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if coverage.get("mode") == "before_after_compare":
            overall = coverage.get("overall") or {}
            metrics["overall.line"] = float(overall.get("generated_avg", 0.0) or 0.0)
            for layer in plan.get("enabled_layers", []):
                layer_meta = (coverage.get("per_layer") or {}).get(str(layer), {})
                generated_meta = layer_meta.get("generated", {}) if isinstance(layer_meta, dict) else {}
                metrics[f"{layer}.line"] = float(generated_meta.get("line_coverage", 0.0) or 0.0)
                function_coverage = generated_meta.get("function_coverage")
                if function_coverage is not None:
                    metrics[f"{layer}.function"] = float(function_coverage)
                branch_coverage = generated_meta.get("branch_coverage")
                if branch_coverage is not None:
                    metrics[f"{layer}.branch"] = float(branch_coverage)
            return metrics

        overall = coverage.get("overall") or {}
        metrics["overall.line"] = float(overall.get("line_coverage", 0.0) or 0.0)
        for layer in plan.get("enabled_layers", []):
            layer_meta = (coverage.get("per_layer") or {}).get(str(layer), {})
            if not isinstance(layer_meta, dict):
                continue
            metrics[f"{layer}.line"] = float(layer_meta.get("line_coverage", 0.0) or 0.0)
            function_coverage = layer_meta.get("function_coverage")
            if function_coverage is not None:
                metrics[f"{layer}.function"] = float(function_coverage)
            branch_coverage = layer_meta.get("branch_coverage")
            if branch_coverage is not None:
                metrics[f"{layer}.branch"] = float(branch_coverage)
        return metrics

    def _update_coverage_progress(self, state, plan: Dict[str, Any], coverage: Dict[str, Any]) -> Dict[str, Any]:
        metrics = self._coverage_progress_metrics(plan, coverage)
        previous_best = getattr(state, "best_coverage_metrics", {}) or {}
        if not isinstance(previous_best, dict):
            previous_best = {}

        epsilon = 1e-6
        improved = False
        regressions: Dict[str, Dict[str, float]] = {}
        for key, value in metrics.items():
            try:
                previous_value = float(previous_best.get(key, -1.0))
            except (TypeError, ValueError):
                previous_value = -1.0
            if float(value) > previous_value + epsilon:
                improved = True
            elif key in previous_best and float(value) < previous_value - epsilon:
                regressions[key] = {
                    "current": float(value),
                    "best": previous_value,
                }

        if improved or not previous_best:
            state.best_coverage_metrics = {
                key: max(float(value), float(previous_best.get(key, -1.0) or -1.0))
                for key, value in metrics.items()
            }
            state.best_coverage_epoch = int(getattr(state, "epoch_current", 1) or 1)
            state.coverage_no_improvement_rounds = 0
        else:
            state.coverage_no_improvement_rounds = int(getattr(state, "coverage_no_improvement_rounds", 0) or 0) + 1

        return {
            "metrics": metrics,
            "improved": improved or not previous_best,
            "regressions": regressions,
            "no_improvement_rounds": state.coverage_no_improvement_rounds,
            "best_metrics": getattr(state, "best_coverage_metrics", {}) or {},
        }

    def _coverage_regression_failures(
        self,
        plan: Dict[str, Any],
        coverage_progress: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        regressions = coverage_progress.get("regressions") or {}
        if not isinstance(regressions, dict) or not regressions:
            return []

        failures: List[Dict[str, Any]] = []
        enabled_layers = [str(layer) for layer in plan.get("enabled_layers", [])]
        for metric_name, meta in regressions.items():
            layer = metric_name.split(".", 1)[0]
            if layer == "overall" or layer not in enabled_layers:
                layer = enabled_layers[0] if enabled_layers else ""
            selected_case = next(
                (
                    case for case in plan.get("cases", [])
                    if isinstance(case, dict) and case.get("layer_id") == layer
                ),
                None,
            )
            block_id = str((selected_case or {}).get("block_id", "HEADER"))
            failures.append(
                {
                    "test": (selected_case or {}).get("test_name_hint", block_id),
                    "block_id": block_id,
                    "file_id": (selected_case or {}).get("file_id", ""),
                    "layer_id": layer,
                    "error_type": "CoverageRegression",
                    "action": "rewrite_block",
                    "note": (
                        f"{metric_name} regressed from {meta.get('best')}% "
                        f"to {meta.get('current')}%; restore or improve the previous best coverage"
                    ),
                }
            )
        return failures[:6]

    @staticmethod
    def _augment_uncovered_digest(
        uncovered_analysis: Optional[Dict[str, Any]],
        layers: List[str],
    ) -> List[Dict[str, Any]]:
        """Compact per-layer uncovered digest for the augment_request handed to the
        planner. Only includes the plateaued layers; keeps a few sample files/lines
        so design_test_plan can target new cases without bloating the prompt."""
        if not uncovered_analysis:
            return []
        wanted = {str(l) for l in layers}
        digest: List[Dict[str, Any]] = []
        for layer_data in uncovered_analysis.get("layers", []):
            if str(layer_data.get("layer")) not in wanted:
                continue
            digest.append({
                "layer": layer_data.get("layer"),
                "total_uncovered": layer_data.get("total_uncovered"),
                "files": [
                    {
                        "source": f.get("source"),
                        "uncovered_count": f.get("uncovered_count"),
                        "sample_lines": (f.get("uncovered_lines") or [])[:8],
                    }
                    for f in (layer_data.get("files", []) or [])[:5]
                ],
            })
        return digest

    def execute(self, state) -> StageResult:
        log_text = str(state.load_artifact("execution_log.txt") or "")
        exit_payload = _parse_exit_payload(state.load_artifact("exit_code.txt"))
        coverage = _load_json_artifact(state, "coverage_summary.json", default={})
        plan = _load_json_artifact(state, "test_plan.json")
        generated_log = str(state.load_artifact("generated_execution_log.txt") or log_text)
        compare_mode = coverage.get("mode") == "before_after_compare" or _coverage_mode(plan) == "before_after_compare"

        analysis = _default_analysis_plan()
        passed = 0
        failed = 0
        errors = 0
        coverage_progress: Dict[str, Any] = {}

        if compare_mode and exit_payload.get("baseline", 0) != 0:
            analysis["status"] = "blocked"
            analysis["stop_recommended"] = True
            analysis["stop_reason"] = "Baseline op_host/op_api UT failed; coverage comparison is unavailable"
            state.auto_stop_reason = analysis["stop_reason"]
            errors = 1
        elif compare_mode and not coverage.get("skipped_layers") and not ((coverage.get("runs") or {}).get("baseline") or {}).get("coverage_valid", True):
            baseline_errors = ((coverage.get("runs") or {}).get("baseline") or {}).get("coverage_errors") or {}
            ordered = _ordered_files(plan)
            first_test_file_id = next((f["file_id"] for f in ordered if f.get("kind") != "cmake"), "")
            first_layer_id = next((f["layer_id"] for f in ordered if f.get("kind") != "cmake"), "")
            analysis["status"] = "failed"
            failed = 1
            errors = 1
            analysis["stop_reason"] = (
                "Baseline op_host/op_api coverage collection failed; "
                f"coverage comparison is unavailable: {baseline_errors}"
            )
            analysis["failures"] = [
                {
                    "test": "coverage_collection",
                    "block_id": "HEADER",
                    "file_id": str(first_test_file_id),
                    "layer_id": str(first_layer_id),
                    "error_type": "CoverageCollectionError",
                    "action": "fix_dependency",
                    "note": f"no coverage data collected; check that tests register and exercise the operator code: {baseline_errors}",
                }
            ]
        elif (compare_mode and exit_payload.get("generated", 0) != 0) or exit_payload.get("overall", 0) != 0:
            failure = self._map_failure(state, plan, generated_log if compare_mode else log_text)
            analysis["failures"] = [failure]
            failed = 1
            if failure["error_type"] in {"CompilationError", "CoverageCollectionError", "CoverageInstrumentationError", "ZeroTestsRegistered"}:
                errors = 1
            analysis["status"] = "failed"
        elif not coverage.get("coverage_valid", True):
            ordered = _ordered_files(plan)
            first_test_file_id = next((f["file_id"] for f in ordered if f.get("kind") != "cmake"), "")
            first_layer_id = next((f["layer_id"] for f in ordered if f.get("kind") != "cmake"), "")
            cov_errors = coverage.get("coverage_errors") or {}
            first_zero = next(
                (v for v in cov_errors.values() if v == "ZeroTestsRegistered"), None
            )
            if first_zero:
                error_type = "ZeroTestsRegistered"
                action = "rewrite_block"
                note = (
                    f"test binary ran 0 tests — attest-generated *_attest.cpp was not compiled "
                    f"into the binary or TEST_F registrations are missing: {cov_errors}"
                )
            else:
                error_type = "CoverageCollectionError"
                action = "fix_dependency"
                note = f"coverage collection is invalid: {cov_errors}"
            failure = {
                "test": "coverage_collection",
                "block_id": "HEADER",
                "file_id": str(first_test_file_id),
                "layer_id": str(first_layer_id),
                "error_type": error_type,
                "action": action,
                "note": note,
            }
            analysis["failures"] = [failure]
            failed = 1
            errors = 1
            analysis["status"] = "failed"
        else:
            coverage_progress = self._update_coverage_progress(state, plan, coverage)
            coverage_failures = self._coverage_failures(state, plan, coverage)
            if not coverage_failures:
                coverage_failures = self._coverage_regression_failures(plan, coverage_progress)
            if coverage_failures:
                analysis["failures"] = coverage_failures
                failed = len(coverage_failures)
                analysis["status"] = "not_fully_passed"
            else:
                analysis["status"] = "success"
                passed = len(plan.get("cases", []))
                stop_policy = _coverage_stop_policy(plan)
                if stop_policy["stop_on_threshold"]:
                    analysis["stop_recommended"] = True
                    if compare_mode:
                        analysis["stop_reason"] = (
                            "Generated op_host/op_api coverage meets the enhance thresholds "
                            "(overall >= 90%, per layer >= 85%) and is not lower than the baseline"
                        )
                    else:
                        analysis["stop_reason"] = "All enabled layers passed and reached the coverage threshold"
                    state.auto_stop_reason = analysis["stop_reason"]
                elif coverage_progress.get("no_improvement_rounds", 0) >= stop_policy["patience_no_improvement"]:
                    # Phase 2: before giving up on a plateau, try ONE round of
                    # planner augmentation for layers still below threshold. The
                    # reviewer (this stage) emits an augment_request that the engine
                    # routes back to design_test_plan instead of stopping.
                    prev_plan = _load_json_artifact(state, "analysis_plan.json", default={})
                    augment_rounds = int(prev_plan.get("augment_rounds", 0) or 0)
                    MAX_AUGMENT = 1
                    layer_threshold = _layer_coverage_threshold(plan)
                    below = [
                        str(layer)
                        for layer in plan.get("enabled_layers", [])
                        if float((coverage.get("per_layer") or {}).get(str(layer), {}).get("line_coverage", 0.0))
                        < layer_threshold
                    ]
                    if augment_rounds < MAX_AUGMENT and below:
                        analysis["augment_rounds"] = augment_rounds + 1
                        analysis["replan_recommended"] = True
                        analysis["augment_request"] = {
                            "layers": below,
                            "reason": "coverage plateau",
                            "uncovered": [],  # backfilled after uncovered_analysis is computed
                        }
                        analysis["stop_reason"] = (
                            f"Coverage plateaued; augmenting test plan for layers {below} "
                            f"(augment round {augment_rounds + 1}/{MAX_AUGMENT})"
                        )
                    else:
                        analysis["augment_rounds"] = augment_rounds
                        analysis["stop_recommended"] = True
                        analysis["stop_reason"] = (
                            "Enhanced coverage did not improve for "
                            f"{coverage_progress['no_improvement_rounds']} consecutive rounds; "
                            "stop at the current best coverage"
                        )
                        state.auto_stop_reason = analysis["stop_reason"]

        analysis["passed"] = passed
        analysis["failed"] = failed
        analysis["errors"] = errors
        if coverage_progress:
            analysis["coverage_progress"] = coverage_progress

        uncovered_analysis: Optional[Dict[str, Any]] = None
        try:
            uncovered_analysis = self._collect_uncovered_code(state, plan, coverage)
        except Exception as e:
            analysis["uncovered_collection_error"] = str(e)[:200]
        if uncovered_analysis and uncovered_analysis.get("total_files", 0) > 0:
            analysis["uncovered_analysis"] = {
                "total_files": uncovered_analysis.get("total_files"),
                "total_uncovered_lines": uncovered_analysis.get("total_uncovered_lines"),
                "layers": [
                    {
                        "layer": layer_data.get("layer"),
                        "total_uncovered": layer_data.get("total_uncovered"),
                        "files": [
                            {
                                "source": f.get("source"),
                                "uncovered_count": f.get("uncovered_count"),
                                "sample_lines": (f.get("uncovered_lines") or [])[:10],
                            }
                            for f in layer_data.get("files", [])
                        ],
                    }
                    for layer_data in uncovered_analysis.get("layers", [])
                ],
            }
            state.save_artifact("uncovered_code.json", _render_json(uncovered_analysis))
        # Phase 2: backfill augment_request uncovered digest now that we have data
        if analysis.get("augment_request"):
            below = analysis["augment_request"].get("layers", [])
            analysis["augment_request"]["uncovered"] = self._augment_uncovered_digest(
                uncovered_analysis, below
            )
        analysis_text = _render_json(analysis)

        md_lines = [
            f"# Ascend UT Analysis - {state.target}",
            "",
            f"- Status: `{analysis['status']}`",
            f"- Passed: `{analysis['passed']}`",
            f"- Failed: `{analysis['failed']}`",
            f"- Errors: `{analysis['errors']}`",
            "",
        ]
        if compare_mode:
            md_lines.append("## Coverage Delta")
            for layer in plan.get("enabled_layers", []):
                meta = (coverage.get("per_layer") or {}).get(str(layer), {})
                baseline_cov = float((meta.get("baseline") or {}).get("line_coverage", 0.0)) if isinstance(meta, dict) else 0.0
                generated_cov = float((meta.get("generated") or {}).get("line_coverage", 0.0)) if isinstance(meta, dict) else 0.0
                delta = float(meta.get("delta_line_coverage", 0.0)) if isinstance(meta, dict) else 0.0
                md_lines.append(
                    f"- `{layer}`: baseline {baseline_cov}% -> generated {generated_cov}% (delta {delta:+.2f}%)"
                )
            md_lines.append("")
        md_lines.append("## Blocks To Fix")
        if analysis["failures"]:
            for item in analysis["failures"]:
                md_lines.append(
                    f"- `{item['block_id']}` ({item['layer_id']}/{item['file_id']}): `{item['action']}` because `{item['error_type']}`"
                )
        else:
            md_lines.append("- None")
        md_lines.extend(
            [
                "",
                f"- stop_recommended: `{analysis['stop_recommended']}`",
                f"- stop_reason: `{analysis['stop_reason'] or 'none'}`",
            ]
        )
        analysis_md = "\n".join(md_lines)

        state.save_artifact("analysis_plan.json", analysis_text)
        state.save_artifact("analysis.md", analysis_md)

        # Save case inventory for next-epoch context (P4)
        try:
            generated_root = self._generated_project_root(state, plan, coverage)
            inventory_blocks: List[Dict[str, Any]] = []
            for file_entry in plan.get("files", []):
                if not isinstance(file_entry, dict):
                    continue
                file_id = str(file_entry.get("file_id", ""))
                fpath = generated_root / str(file_entry.get("path", ""))
                for blk in build_block_entries(fpath):
                    inventory_blocks.append({
                        "block_id": blk["block_id"],
                        "file_id": file_id,
                        "status": blk["status"],
                    })
            case_inventory = {
                "epoch": getattr(state, "epoch_current", 1),
                "generated_blocks": inventory_blocks,
            }
            state.save_artifact("case_inventory.json", _render_json(case_inventory))
        except Exception:
            pass

        try:
            epoch_num = int(getattr(state, "epoch_current", 1) or 1)
            manifest_candidates = [
                state.artifacts_dir / "generate_code" / f"v{epoch_num}_generation_manifest.json",
                state.artifacts_dir / state.current_stage / f"v{epoch_num}_generation_manifest.json",
            ]
            for manifest_path in manifest_candidates:
                if manifest_path.exists():
                    state.save_epoch_snapshot(epoch_num, manifest_path)
                    break
        except Exception:
            pass

        return StageResult(
            True,
            {"analysis_plan.json": analysis_text, "analysis.md": analysis_md},
            message=f"Analysis status: {analysis['status']}",
        )


class AscendGenerationAgentLoopStage(AscendBaseStage):
    """
    Continuous agent loop that fuses generate_code + execute_tests + analyze_results
    into a single LLM session. The LLM autonomously iterates: write → compile → run
    → check coverage → fill gaps → repeat until satisfied or turns exhausted.

    Replaces stages 4+5+6 in the `ascend_ut_continuous` profile. Produces the same
    artifact set as the three separate stages so that generate_report (stage 7) works
    without any modification.
    """

    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="generate_code",
            display_name="Generation Agent Loop (Continuous)",
            description="Single continuous LLM session: generate + compile + coverage + analyze",
            prompt_template="",
            input_artifacts=[
                "operator_context.json", "requirements.md", "test_plan.json",
                "analysis_plan.json",
            ],
            output_artifacts=[
                "generation_manifest.json",
                "execution_log.txt",
                "coverage_summary.json",
                "analysis_plan.json",
                "analysis.md",
            ],
            tools=[
                "exec_command",
            ],
            allow_skip=False,
        )
        # Mark this stage as an epoch boundary so engine.py's epoch loop fires on it
        self.is_epoch_boundary = True
        # Parallel per-layer generator agents (1 = sequential single-session, legacy)
        self.workers = 1

    def set_workers(self, n: int) -> None:
        self.workers = max(1, int(n or 1))

    # ------------------------------------------------------------------
    # Build commands helpers (reuse AscendBaseStage infrastructure)
    # ------------------------------------------------------------------

    def _build_layer_cmd_section(
        self,
        plan: Dict[str, Any],
        project_root: Path,
    ) -> str:
        build_plan = plan.get("build_plan") or {}
        if not isinstance(build_plan, dict):
            return ""
        lines: List[str] = ["## Build and coverage commands per layer\n"]
        for layer, cmds in build_plan.items():
            if not isinstance(cmds, dict):
                continue
            compile_cmd = (cmds.get("compile_cmd") or "").replace("{project_root}", str(project_root))
            coverage_cmd = (cmds.get("coverage_cmd") or "").replace("{project_root}", str(project_root))
            if compile_cmd:
                lines.append(f"**{layer}** compile+run:  `{compile_cmd}`")
            if coverage_cmd:
                lines.append(f"**{layer}** coverage:      `{coverage_cmd}`")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_agent_loop_prompt(
        self,
        state,
        plan: Dict[str, Any],
        context: Dict[str, Any],
        project_root: Path,
        slim_context: Dict[str, Any],
        analysis_plan: Dict[str, Any],
        prev_error_context: str,
        case_inventory_context: str,
        provider,
        generation_mode: str,
        coverage_mode: str,
        infra_context_snippet: str = "",
    ) -> str:
        epoch = getattr(state, "epoch_current", 1)
        epoch_total = getattr(state, "epoch_total", 1)
        enabled_layers = plan.get("enabled_layers", [])
        files_info = [
            f"  - {f.get('path', '?')} (layer={f.get('layer_id','?')}, kind={f.get('kind','?')})"
            for f in plan.get("files", [])
            if isinstance(f, dict)
        ]
        cmd_section = self._build_layer_cmd_section(plan, project_root)
        skill_text = ""
        for layer in enabled_layers:
            fewshot = _get_few_shot_examples(str(layer))
            if fewshot:
                skill_text += f"\n{fewshot}\n"
            pkt = provider.get_stage_packet("generate_code", str(layer), generation_mode=generation_mode)
            if pkt:
                skill_text += f"\n### Skill [{layer}]\n{pkt}\n"

        augment_text = ""
        directive = plan.get("augment_directive") or {}
        if directive:
            augment_text = (
                f"\n## ⚠️ AUGMENTATION DIRECTIVE (reviewer flagged layers "
                f"{directive.get('layers', [])} as plateaued)\n"
                f"Coverage stalled below threshold for these layers. Focus this epoch on the "
                f"uncovered lines below — read the source around each, understand the branch, and ADD "
                f"new TEST_F cases exercising those exact paths. Do NOT rewrite already-passing cases.\n"
                f"```json\n{json.dumps(directive.get('uncovered', []), ensure_ascii=False, indent=2)[:2500]}\n```\n"
            )

        return f"""You are an Ascend C++ unit-test generation agent with FULL autonomy.
Epoch {epoch}/{epoch_total}. Operator: {slim_context.get('op_name', '?')}

## Your mission
Generate thorough GTest unit-tests that maximise LINE coverage of the operator source code.
Work completely within this conversation — write code, compile, run, check coverage, fill gaps,
repeat — until you are satisfied or turns are exhausted.

## Files to write (project root: {project_root})
{chr(10).join(files_info)}

{cmd_section}

## Mandatory workflow (loop until coverage stabilises or turns run out)

1. **For each file** (in order of layer priority: op_api first, then op_host):
   a. Read the current block index to see what's already filled vs placeholder.
   b. Fill every placeholder block with meaningful test cases.
      - Use `exec_command` with heredoc to write file content: `cat > path/file.cpp << 'EOF'\n...content...\nEOF`
      - Use `sed` to replace or update existing blocks.
   c. Immediately compile+run: `exec_command(cmd="<compile_cmd_for_layer>")`
   d. If compile fails: read the error (`cat` log), reason, read API source (`cat`, `grep`), fix, recompile.
   e. If runtime abort/SIGSEGV: same loop. Do NOT move on until tests run cleanly.

2. **After all files are done**, run coverage for each enabled layer and read the output (`cat` coverage files, `grep` for metrics).

3. **Identify uncovered lines** — read the operator source around each uncovered line (`cat -n`, `sed -n`),
   understand the branch condition, write a new TEST_F that exercises that path.

4. **Recompile and re-run** after each gap-filling round. Repeat steps 2-4 until
   coverage no longer improves or you have fewer than 20 turns remaining.

5. **At the very end**, output a JSON block (surrounded by ```json ... ```) with this schema:
```json
{{
  "status": "success" | "partial" | "compile_failed",
  "coverage_per_layer": {{"op_api": 0.0, "op_host": 0.0}},
  "failures": [],
  "stop_reason": "coverage_plateau | turns_exhausted | compile_error"
}}
```

## Key rules
- You have ONLY `exec_command`. Use bash commands (`cat`, `grep`, `sed`, `find`, `head`, `tail`, `awk`, heredocs) for ALL file operations.
- NEVER guess an API signature. Always `grep` / `cat` the implementation before using a type.
- Keep every test traceable to its BLOCK_ID marker.
- DO NOT regenerate already-filled blocks unless you are clearly improving them. DO NOT overwrite a non-empty file from scratch with fewer TEST_F cases than it already has; instead, edit specific blocks only.
- Each `exec_command` output is your ground-truth — trust it over your prior assumptions.
- ⚠️ **CRITICAL — STUB REPLACEMENT** (rule 10): Every `// ==== BLOCK: CASE_NN START ====` block has been pre-populated with a STUB that looks like:
    ```cpp
    // STUB: replace this block with real test for case_type=...
    TEST(AttestStubs, CASE_NN_...) {{ GTEST_SKIP() << "STUB — case_type=..."; }}
    ```
    Your job is to **replace every STUB with a real, working TEST_F** before finishing. If you CANNOT complete a stub, **LEAVE IT AS GTEST_SKIP()** — do NOT change it to FAIL() or any other assertion. GTEST_SKIP() gracefully skips the test (no impact on other tests), while FAIL() will crash the entire test suite and ruin coverage.
    Before concluding the session, run `grep -n 'STUB: replace' <file>` on each test file — if ANY STUBs remain, go back and replace each one. The replacement must:
    1. Call the real operator API (e.g. `OP_API_UT_EXPECT`, or build infershape inputs for op_host)
    2. Use the test-fixture class declared in the HEADER block (e.g. `TEST_F(MyTestClass, ...)`), not `TEST(AttestStubs, ...)`
    3. Exercise the operator code path described by `case_type` (e.g. `infershape_basic` → basic shape inference)

## Operator context
```json
{json.dumps(slim_context, ensure_ascii=False, indent=2)[:8000]}
```

## Test plan (files + cases)
```json
{json.dumps(plan, ensure_ascii=False, indent=2)[:3000]}
```

## Prior analysis plan (from previous epoch, if any)
```json
{json.dumps(analysis_plan, ensure_ascii=False, indent=2)[:2000]}
```
{infra_context_snippet}{prev_error_context}{case_inventory_context}
{augment_text}
{skill_text}
Begin now. Start with the first file."""

    # ------------------------------------------------------------------
    # Post-session: parse summary JSON from LLM response text
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_agent_summary(llm_text: str) -> Dict[str, Any]:
        """Extract the final JSON summary block from LLM output."""
        import re
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_text or "")
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    # Parallel per-layer generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_files(plan: Dict[str, Any], layer: str) -> List[Dict[str, Any]]:
        """Files belonging to a single layer (excludes cmake scaffolding)."""
        return [
            f
            for f in plan.get("files", [])
            if isinstance(f, dict)
            and str(f.get("layer_id", "")) == str(layer)
            and str(f.get("kind", "")) != "cmake"
        ]

    def _per_layer_turn_limit(self) -> int:
        """Split the single-session 300-turn budget across parallel layer agents.

        Keeps the total token/turn budget roughly constant: --workers>1 trades
        wall-clock for parallelism, not budget.
        """
        return max(60, 300 // max(1, self.workers))

    @staticmethod
    def _isolated_build_prefix(project_root: Path, layer: str, op_name: str) -> str:
        """Env prefix that pins a layer's LLM-driven compiles to its own build dir.

        In the continuous loop the LLM itself issues exec_command with the compile
        command, so the framework prefix in _run_for_root never applies. We bake
        this exact prefix into the command strings shown in the layer prompt.
        Label matches _collect_parallel_coverage's `generated_{layer}` so the build
        dir names line up.
        """
        label = f"generated_{layer}"
        build_dir = project_root / f"build_{label}_{op_name}"
        build_out_dir = project_root / f"build_out_{label}_{op_name}"
        return (
            f"BUILD_PATH={shlex.quote(str(build_dir))} "
            f"BUILD_OUT_PATH={shlex.quote(str(build_out_dir))} "
        )

    def _build_single_layer_prompt(
        self,
        state,
        plan: Dict[str, Any],
        layer: str,
        project_root: Path,
        slim_context: Dict[str, Any],
        analysis_plan: Dict[str, Any],
        prev_error_context: str,
        case_inventory_context: str,
        provider,
        generation_mode: str,
        infra_context_snippet: str = "",
    ) -> str:
        """Layer-scoped variant of _build_agent_loop_prompt for one parallel agent.

        Restricts the file list and build commands to a single layer, and bakes the
        isolated BUILD_PATH prefix into the compile/coverage command strings.
        """
        epoch = getattr(state, "epoch_current", 1)
        epoch_total = getattr(state, "epoch_total", 1)
        op_name = slim_context.get("op_name", "?")
        cpp_files = self._layer_files(plan, layer)
        files_info = [
            f"  - {f.get('path', '?')} (layer={f.get('layer_id','?')}, kind={f.get('kind','?')})"
            for f in cpp_files
        ]
        cmake_file = next(
            (f for f in plan.get("files", [])
             if isinstance(f, dict)
             and str(f.get("layer_id", "")) == str(layer)
             and str(f.get("kind", "")) == "cmake"),
            None,
        )
        if cmake_file and (project_root / str(cmake_file.get("path", ""))).exists():
            files_info.append(
                f"  - {cmake_file.get('path', '?')} (layer={layer}, kind=cmake) [BUILD REGISTRATION — see Rule 15]"
            )
        build_prefix = self._isolated_build_prefix(project_root, layer, str(op_name))
        cmds = (plan.get("build_plan") or {}).get(layer, {})
        compile_cmd = (cmds.get("compile") or cmds.get("compile_cmd") or "").replace(
            "{project_root}", str(project_root)
        )
        coverage_cmd = (cmds.get("coverage") or cmds.get("coverage_cmd") or "").replace(
            "{project_root}", str(project_root)
        )
        cmd_lines = [f"## Build and coverage commands for layer `{layer}`\n"]
        if compile_cmd:
            cmd_lines.append(f"**compile+run:**  `{build_prefix}{compile_cmd}`")
        if coverage_cmd:
            cmd_lines.append(f"**coverage:**     `{build_prefix}{coverage_cmd}`")
        cmd_section = "\n".join(cmd_lines) + "\n"

        pkt = provider.get_stage_packet("generate_code", str(layer), generation_mode=generation_mode)
        fewshot = _get_few_shot_examples(str(layer))
        skill_text = (f"\n{fewshot}\n" if fewshot else "") + (f"\n### Skill [{layer}]\n{pkt}\n" if pkt else "")

        augment_text = ""
        directive = plan.get("augment_directive") or {}
        if directive and str(layer) in [str(l) for l in directive.get("layers", [])]:
            uncovered = [u for u in directive.get("uncovered", []) if str(u.get("layer")) == str(layer)]
            augment_text = (
                f"\n## ⚠️ AUGMENTATION DIRECTIVE (reviewer flagged `{layer}` as plateaued)\n"
                f"Coverage stalled below threshold. Focus this epoch on the uncovered lines below — "
                f"read the source around each, understand the branch, and ADD new TEST_F cases that "
                f"exercise those exact paths. Do NOT rewrite already-passing cases.\n"
                f"```json\n{json.dumps(uncovered, ensure_ascii=False, indent=2)[:2500]}\n```\n"
            )

        return f"""You are an Ascend C++ unit-test generation agent with FULL autonomy.
Epoch {epoch}/{epoch_total}. Operator: {op_name}. **You own ONLY the `{layer}` layer.**

## Your mission
Generate thorough GTest unit-tests that maximise LINE coverage of the `{layer}` source code.
Work only on the files listed below. Another agent owns the other layer(s) concurrently —
do NOT touch files outside your layer. Write code, compile, run, check coverage, fill gaps,
repeat — until you are satisfied or turns are exhausted.

## Files to write (project root: {project_root})
{chr(10).join(files_info)}

{cmd_section}
## CRITICAL: isolated build directory
Always run the compile/coverage command with the EXACT `BUILD_PATH=... BUILD_OUT_PATH=...`
prefix shown above. NEVER change BUILD_PATH or run a bare `bash build.sh` — a parallel agent
is compiling the other layer, and a shared build directory would corrupt both builds.

## Mandatory workflow (loop until coverage stabilises or turns run out)
1. For each file: read the current block index (`cat`), fill every placeholder block with meaningful
   test cases using `cat > file << 'EOF'` heredoc or `sed` for updates.
2. Immediately compile+run with the prefixed command above. If compile fails: read the error (`cat` log),
   read the API source (`cat`/`grep`), fix (`sed`), recompile. Do NOT move on until tests run cleanly.
3. Run coverage (prefixed command), read the output (`cat`/`grep`), identify uncovered lines, add TEST_F cases
   that exercise those paths. Recompile. Repeat until coverage plateaus or <20 turns remain.
4. At the very end, output a JSON block (```json ... ```) with this schema:
```json
{{
  "status": "success" | "partial" | "compile_failed",
  "coverage_per_layer": {{"{layer}": 0.0}},
  "failures": [],
  "stop_reason": "coverage_plateau | turns_exhausted | compile_error"
}}
```

## Key rules
- NEVER guess an API signature. Always `cat` / `grep` the implementation first.
- Keep every test traceable to its BLOCK_ID marker.
- DO NOT regenerate already-filled blocks unless clearly improving them. DO NOT overwrite a non-empty file from scratch with fewer TEST_F cases than it already has; edit specific blocks only.
- Each exec_command output is your ground-truth.
- ⚠️ **CRITICAL — STUB REPLACEMENT** (rule 10): Every `// ==== BLOCK: CASE_NN START ====` block has been pre-populated with a STUB `TEST(AttestStubs, ...)`. Your job is to **replace every STUB with a real, working TEST_F** that:
    1. Calls the real operator API (e.g. `OP_API_UT_EXPECT` for op_api, or build infershape inputs for op_host)
    2. Uses the test-fixture class declared in the HEADER block (e.g. `TEST_F(MyTestClass, ...)`), not `TEST(AttestStubs, ...)`
    3. Exercises the operator code path described by `case_type`
  **NEVER replace GTEST_SKIP() with FAIL()** — if you cannot fill a stub, LEAVE the GTEST_SKIP() in place. FAIL() will crash the entire test suite and produce 0% coverage.
  Before concluding, run `grep -c 'STUB: replace' <file>` on your output — if ANY STUBs remain, go back and replace each.
- **CRITICAL — CMakeLists.txt registration (Rule 15):** The Ascend build system uses GLOB patterns to auto-discover test files: op_host matches `test_*_infershape.cpp` / `test_*_tiling*.cpp`, op_api matches `test_aclnn_*.cpp`. Your `*_attest.cpp` files already match these patterns and will be picked up automatically. If a `CMakeLists.txt` exists in your layer's test directory, you may need to update its FOOTER block to register new files. If no `CMakeLists.txt` exists, DO NOT create one — the build system handles discovery via GLOB.
- NEVER rewrite an existing `CMakeLists.txt` HEADER block — it contains framework-required preamble. Only touch the FOOTER block to add source registration lines if needed.

## Operator context
```json
{json.dumps(slim_context, ensure_ascii=False, indent=2)[:8000]}
```

## Prior analysis plan (from previous epoch, if any)
```json
{json.dumps(analysis_plan, ensure_ascii=False, indent=2)[:2000]}
```
{infra_context_snippet}{prev_error_context}{case_inventory_context}
{augment_text}
{skill_text}
Begin now. Start with the first file of the `{layer}` layer."""

    def _run_layer_session(
        self,
        state,
        plan: Dict[str, Any],
        layer: str,
        project_root: Path,
        slim_context: Dict[str, Any],
        analysis_plan: Dict[str, Any],
        prev_error_context: str,
        case_inventory_context: str,
        provider,
        generation_mode: str,
        infra_context_snippet: str = "",
    ):
        """Worker body run in a thread: one layer-scoped LLM generation session.

        Never calls save_artifact and never patches build.sh — only generates.
        """
        prompt = self._build_single_layer_prompt(
            state, plan, layer, project_root, slim_context, analysis_plan,
            prev_error_context, case_inventory_context, provider, generation_mode,
            infra_context_snippet=infra_context_snippet,
        )
        res = self._run_llm_session(
            state, prompt, project_root, turn_limit=self._per_layer_turn_limit()
        )
        return layer, res, self._parse_agent_summary(res.message or "")

    def _collect_parallel_coverage(
        self,
        state,
        plan: Dict[str, Any],
        project_root: Path,
        enabled_layers: List[str],
    ) -> Dict[str, Any]:
        """After parallel generation, run per-layer coverage builds SEQUENTIALLY
        on the main thread and merge into one coverage_summary.json.

        Serializing the coverage builds avoids the build.sh patch/restore race;
        each layer still compiles into its own isolated build dir (label
        `generated_{layer}`), matching the BUILD_PATH baked into the layer prompt.
        """
        exec_stage = AscendExecutionStage(self.llm, self.tool_runner)
        merged_per_layer: Dict[str, Dict[str, Any]] = {}
        logs: List[str] = []
        exit_code = 0
        for layer in enabled_layers:
            plan_slice = copy.deepcopy(plan)
            plan_slice["enabled_layers"] = [layer]
            bp = plan_slice.get("build_plan") or {}
            plan_slice["build_plan"] = {
                k: v for k, v in bp.items() if k == layer or k == "_combined"
            }
            # Keep _combined: the combined coverage command builds both layers
            # together, ensuring CANN runtime libraries link properly (op_api
            # depends on symbols from op_host build). Per-layer coverage is
            # extracted from the combined build output.
            try:
                run = exec_stage._run_for_root(
                    plan_slice, project_root, use_coverage=True,
                    label=f"generated_{layer}", state_obj=state,
                )
                layer_summary = (run.get("summary") or {}).get("per_layer", {}).get(str(layer))
                if layer_summary is not None:
                    merged_per_layer[str(layer)] = layer_summary
                logs.append(run.get("log_text", ""))
                if int(run.get("exit_code", 0)) != 0:
                    exit_code = 1
            except Exception as exc:  # per-layer failure must not sink the other layer
                merged_per_layer[str(layer)] = exec_stage._default_layer_summary(
                    threshold=_layer_coverage_threshold(plan),
                    coverage_valid=False,
                    error_reason=f"parallel coverage build failed: {exc}",
                )
                logs.append(f"=== generated_{layer} coverage build raised: {exc} ===")
                exit_code = 1

        summary = exec_stage._build_run_summary(plan, project_root, merged_per_layer, exit_code)
        combined_log = "\n\n".join(l for l in logs if l).strip()
        exit_payload = {"overall": int(exit_code)}
        coverage_text = _render_json(summary)
        state.save_artifact("execution_log.txt", combined_log)
        state.save_artifact("exit_code.txt", _render_json(exit_payload))
        state.save_artifact("coverage_summary.json", coverage_text)
        prev_epoch_content = _extract_execution_summary(combined_log, max_chars=8000)
        state.save_artifact("prev_epoch_error_log.txt", prev_epoch_content)
        return {
            "summary": summary,
            "coverage_summary.json": coverage_text,
            "execution_log.txt": combined_log,
            "exit_code.txt": _render_json(exit_payload),
        }

    def _default_layer_summary(self, *args, **kwargs):
        """Delegate to AscendExecutionStage's helper for consistent schema."""
        return AscendExecutionStage._default_layer_summary(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # LLM session helpers (copied from AscendCodeGenStage)
    # ------------------------------------------------------------------

    def _tool_schemas(self) -> List[Dict[str, Any]]:
        allowed = set(self.config.tools)
        return [
            schema
            for schema in self.tool_runner.registry.to_llm_schema()
            if schema["function"]["name"] in allowed
        ]

    def _run_llm_session(self, state, prompt: str, project_root: Path, turn_limit: int = 70) -> StageResult:
        messages = [{"role": "user", "content": prompt}]
        append_message(
            session_id=getattr(state, "workflow_id", "workflow"),
            role="user",
            content={"stage": self.config.name, "prompt": prompt},
            workspace=str(state.workspace),
            stage=self.config.name,
        )
        ctx = ToolContext(cwd=str(project_root), auto_approve=True)
        tool_schemas = self._tool_schemas()

        STAGNATION_LIMIT = 8
        _stagnation_streak = 0
        _last_tool_calls_signature = ""

        _api400_retries = 0

        for turn in range(turn_limit):
            _made_progress = False
            _api400_response = None
            while _api400_response is None:
                try:
                    _api400_response = self.llm.chat(messages, tools=tool_schemas)
                except Exception as exc:
                    exc_str = str(exc)
                    if "400" in exc_str and _api400_retries < 3:
                        _api400_retries += 1
                        warning_msg = (
                            f"⚠️ API returned HTTP 400 (arguments too large). "
                            f"Retry {_api400_retries}/3. "
                            f"Please split your output into smaller chunks: use `cat >` heredocs for the first ~100 lines, "
                            f"then use `cat >>` or `sed` for remaining content in batches of ~100 lines each. "
                            f"Keep each `exec_command` under 5000 characters."
                        )
                        messages.append({"role": "user", "content": warning_msg})
                        _made_progress = True
                        break
                    return StageResult(False, {}, error=f"LLM call failed: {exc}")
            if _api400_response is None:
                continue
            response = _api400_response

            assistant_msg = {
                "role": "assistant",
                "content": response.content,
                "reasoning_content": getattr(response, "reasoning_content", "") or "",
            }
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)
            append_message(
                session_id=getattr(state, "workflow_id", "workflow"),
                role="assistant",
                content=assistant_msg,
                workspace=str(state.workspace),
                stage=self.config.name,
            )

            if not response.has_tool_calls():
                return StageResult(True, {}, message=response.content or "Generated file blocks")

            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    warning_msg = (
                        "⚠️ Your previous tool call was rejected — the arguments were truncated (invalid JSON). "
                        "Please split your output into smaller chunks: use `cat >` heredocs for the first ~100 lines, "
                        "then use `cat >>` or `sed` for remaining content in batches of ~100 lines each. "
                        "Keep each `exec_command` under 5000 characters."
                    )
                    messages.append({"role": "user", "content": warning_msg})
                    _made_progress = True
                    continue
                tool_result = self.tool_runner.execute(tool_name, tool_args, ctx)
                if tool_result.ok and tool_name in {
                    "exec_command", "write_file", "replace_in_file", "replace_block", "append_to_file",
                }:
                    _made_progress = True
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result.output if tool_result.ok else tool_result.error or "",
                }
                messages.append(tool_msg)
                append_message(
                    session_id=getattr(state, "workflow_id", "workflow"),
                    role="tool",
                    content=tool_msg,
                    workspace=str(state.workspace),
                    stage=self.config.name,
                )

            current_sig = json.dumps(
                [
                    {"fn": tc["function"]["name"], "args": tc["function"]["arguments"]}
                    for tc in (response.tool_calls or [])
                ],
                sort_keys=True,
            )
            if not _made_progress and current_sig == _last_tool_calls_signature:
                _stagnation_streak += 1
            else:
                _stagnation_streak = 0
            _last_tool_calls_signature = current_sig

            if _stagnation_streak >= STAGNATION_LIMIT:
                print(
                    f"\n  ⚠️  Early stop: {STAGNATION_LIMIT} consecutive iterations"
                    " without progress; exiting tool-calling loop"
                )
                return StageResult(
                    success=False,
                    outputs={},
                    error=(
                        f"Early stop: {STAGNATION_LIMIT} consecutive iterations"
                        " without file modifications"
                    ),
                )

            if turn > 0 and turn % 100 == 0 and len(messages) > 125:
                messages = _compress_old_messages(messages, keep_recent=60)

        return StageResult(False, {}, error="Maximum tool-calling iterations reached")

    # ------------------------------------------------------------------
    # Repository snapshot (copied from AscendCodeGenStage)
    # ------------------------------------------------------------------

    def _prepare_generated_repo(self, state, context: Dict[str, Any]) -> Path:
        target_root = _generated_repo_root(state)
        source_root = Path(state.project_root)
        source_git = source_root / ".git"
        target_git = target_root / ".git"
        if target_root.exists():
            if _is_enhance_mode(context) and source_git.exists() and not target_git.exists():
                shutil.rmtree(target_root)
            else:
                _patch_build_sh_for_isolation(target_root)
                return target_root

        ignore = shutil.ignore_patterns(
            ".attest",
            "__pycache__",
            ".pytest_cache",
            "build",
            "*.pyc",
        )
        shutil.copytree(source_root, target_root, ignore=ignore)
        if _is_enhance_mode(context) and source_git.exists() and not target_git.exists():
            raise RuntimeError(f"Generated snapshot is missing git metadata: {target_git}")

        _patch_build_sh_for_isolation(target_root)
        return target_root

    def _find_source_test_file(self, project_root: Path, attest_path: Path, op_name: str) -> Optional[Path]:
        """Find corresponding source test file for attTest-generated test.
        
        Plan A: When we need to create test_xxx_attest.cpp but it doesn't exist,
        look for source test file (e.g., test_xxx.cpp or test_xxx.cpp.bak) to copy instead.
        This reuses existing test structure and only modifies test body.
        """
        attest_dir = attest_path.parent
        
        # Pattern: test_<op>_attest.cpp -> test_<op>.cpp or test_<op>.cpp.bak
        attest_name = attest_path.name
        if not attest_name.endswith("_attest.cpp"):
            return None
        
        base_name = attest_name.replace("_attest.cpp", "")
        candidates = [
            attest_dir / f"{base_name}.cpp",
            attest_dir / f"{base_name}.cpp.bak",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        # Fallback: any .cpp file with op_name that's not attTest
        for f in attest_dir.glob(f"*{op_name}*.cpp"):
            if "_attest" not in f.name:
                return f

        # Broad fallback: any test_*.cpp that's not _attest
        if attest_dir.exists():
            candidates = sorted([
                f for f in attest_dir.glob("test_*.cpp")
                if not f.name.endswith("_attest.cpp")
            ])
            if candidates:
                return candidates[0]

        return None

    def _find_aclnn_header(self, project_root: Path, op_name: str) -> Optional[str]:
        """Find the aclnn header relative include path for the operator."""
        math_dir = project_root / "math" / op_name
        if not math_dir.exists():
            return None
        headers = sorted(math_dir.rglob("aclnn_*.h"))
        if not headers:
            return None
        best = None
        for h in headers:
            if op_name.replace("_", "") in h.stem.replace("_", ""):
                best = h
                break
        if best is None:
            best = headers[0]
        rel = best.relative_to(math_dir)
        depth = 3
        return "/".join([".."] * depth + list(rel.parts))

    def _scan_aclnn_functions(self, project_root: Path, op_name: str) -> Tuple[str, str]:
        """Scan all aclnn_*.h headers under math/<op_name>/ and extract real function names.

        Returns (main_aclnn_func, header_relative_path) or ("", "") if none found.
        """
        math_dir = project_root / "math" / op_name
        if not math_dir.exists():
            return "", ""
        headers = sorted(math_dir.rglob("aclnn_*.h"))
        if not headers:
            return "", ""

        candidates = []

        for h in headers:
            try:
                text = h.read_text(encoding="utf-8", errors="ignore")
                func_names = re.findall(r'\b(aclnn[A-Za-z0-9]+?)\s*\(', text)
                non_ws = [f for f in func_names if "GetWorkspaceSize" not in f and "Inplace" not in f]
                ws_funcs = [f for f in func_names if "GetWorkspaceSize" in f]

                main_func = non_ws[0] if non_ws else (
                    ws_funcs[0].replace("GetWorkspaceSize", "") if ws_funcs else None
                )
                if not main_func:
                    continue

                sig_match = re.search(
                    r'\b' + re.escape(main_func) + r'GetWorkspaceSize\s*\(([^)]+)\)',
                    text,
                )
                if sig_match:
                    sig = sig_match.group(1)
                    tensor_params = re.findall(r'\bconst\s+aclTensor\s*\*', sig)
                    if len(tensor_params) != 1:
                        continue

                rel_path = h.relative_to(math_dir)
                rel_path_str = "/".join(rel_path.parts)

                if op_name.replace("_", "") in h.stem.replace("_", ""):
                    candidates.insert(0, (main_func, rel_path_str))
                else:
                    candidates.append((main_func, rel_path_str))
            except Exception:
                continue

        if not candidates:
            return "", ""

        best_func, best_include = candidates[0]
        return best_func, best_include

    @staticmethod
    def _op_has_op_api_dir(project_root: Path, op_name: str) -> bool:
        """Check if operator has an op_api directory (aclnn-based API).
        Some aclnn_exclude operators nest op_api under op_host/op_api/."""
        return (project_root / "math" / op_name / "op_api").is_dir() or \
               (project_root / "math" / op_name / "op_host" / "op_api").is_dir()

    @staticmethod
    def _to_pascal_case(snake: str) -> str:
        return "".join(part.capitalize() for part in snake.split("_") if part)

    def _generate_cpp_boilerplate(self, layer_id: str, op_name: str, project_root: Path) -> str:
        """Plan B: generate C++ test boilerplate when no source file exists."""
        if layer_id == "op_api":
            if not self._op_has_op_api_dir(project_root, op_name):
                return ""
            real_func, header_rel = self._scan_aclnn_functions(project_root, op_name)
            if not real_func:
                return ""
            include_depth = 3
            inc_path = "/".join([".."] * include_depth + header_rel.split("/"))
            class_name = f"{op_name}_test"

            ws_func = f"{real_func}GetWorkspaceSize"
            math_dir = project_root / "math" / op_name
            tensor_input_count = 1
            has_scalar = False
            has_aclscalar = False
            for h in sorted(math_dir.rglob("aclnn_*.h")):
                try:
                    text = h.read_text(encoding="utf-8", errors="ignore")
                    sig_match = re.search(
                        r'\b' + re.escape(ws_func) + r'\s*\(([^)]+)\)', text
                    )
                    if not sig_match:
                        continue
                    sig = sig_match.group(1)
                    tensor_params = len(re.findall(r'\bconst\s+aclTensor\s*\*', sig))
                    has_int64 = bool(re.findall(r'\bint64_t\b', sig))
                    has_aclscalar_param = bool(re.findall(r'\baclScalar\s*\*', sig))
                    if tensor_params > 0:
                        tensor_input_count = tensor_params
                        has_scalar = has_int64
                        has_aclscalar = has_aclscalar_param
                        break
                except Exception:
                    continue

            input_descs = "self_desc"
            input_decls = "  auto self_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);"
            for i in range(1, tensor_input_count):
                name = "other_desc" if i == 1 else f"extra{i}_desc"
                input_decls += f"\n  auto {name} = TensorDesc({{2, 3, 4, 5}}, ACL_FLOAT, ACL_FORMAT_ND);"
                input_descs += f", {name}"
            if has_aclscalar:
                input_decls += "\n  int64_t scalar_value = 1;\n  auto scalar_desc = ScalarDesc(scalar_value);"
                input_descs += ", scalar_desc"
            elif has_scalar:
                input_decls += "\n  int64_t scalar_value = 1;"
                input_descs += ", scalar_value"

            has_aclscalar_include = (
                '#include "op_api_ut_common/scalar_desc.h"\n' if has_aclscalar else ""
            )

            return (
                "#include <array>\n"
                "#include <vector>\n"
                '#include "gtest/gtest.h"\n'
                f'#include "{inc_path}"\n'
                '#include "op_api_ut_common/op_api_ut.h"\n'
                '#include "op_api_ut_common/tensor_desc.h"\n'
                f'{has_aclscalar_include}'
                "\n"
                "using namespace std;\n"
                "\n"
                f"class {class_name} : public testing::Test {{\n"
                " protected:\n"
                f"  static void SetUpTestCase() {{ cout << \"{op_name}_test SetUp\" << endl; }}\n"
                f"  static void TearDownTestCase() {{ cout << \"{op_name}_test TearDown\" << endl; }}\n"
                "};\n"
                "\n"
                f"TEST_F({class_name}, case_default_float32) {{\n"
                f"{input_decls}\n"
                "  auto out_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);\n"
                f"  auto ut = OP_API_UT({real_func}, INPUT({input_descs}), OUTPUT(out_desc));\n"
                "  uint64_t workspace_size = 0;\n"
                "  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);\n"
                "  EXPECT_EQ(aclRet, ACL_SUCCESS);\n"
                "}\n"
            )
        elif layer_id == "op_host":
            pascal = self._to_pascal_case(op_name)
            class_name = f"{pascal}InferShape"
            return (
                "#include <gtest/gtest.h>\n"
                "#include <iostream>\n"
                '#include "infershape_context_faker.h"\n'
                '#include "base/registry/op_impl_space_registry_v2.h"\n'
                "\n"
                f"class {class_name} : public testing::Test {{\n"
                " protected:\n"
                f"  static void SetUpTestCase() {{ std::cout << \"{class_name} SetUp\" << std::endl; }}\n"
                f"  static void TearDownTestCase() {{ std::cout << \"{class_name} TearDown\" << std::endl; }}\n"
                "};\n"
                "\n"
                "static std::vector<int64_t> ToVector(const gert::Shape& shape) {\n"
                "  size_t n = shape.GetDimNum();\n"
                "  std::vector<int64_t> v(n, 0);\n"
                "  for (size_t i = 0; i < n; i++) v[i] = shape.GetDim(i);\n"
                "  return v;\n"
                "}\n"
            )
        return ""

    @staticmethod
    def _case_stub_lines(block_id: str, case_type: str, comment_style: str) -> List[str]:
        safe_type = re.sub(r"[^A-Za-z0-9_]", "_", str(case_type))[:40] or "stub"
        test_name = f"{block_id}_{safe_type}"
        return [
            start_marker(block_id, comment_style),
            f"// STUB: replace this block with real test for case_type={case_type}",
            f"TEST(AttestStubs, {test_name}) {{",
            f'    GTEST_SKIP() << "STUB — case_type=\\"{case_type}\\". Replace with real test body.";',
            f"}}",
            end_marker(block_id, comment_style),
        ]

    def _ensure_skeleton(self, project_root: Path, file_entry: Dict[str, Any], file_cases: List[Dict[str, Any]]) -> bool:
        path = project_root / str(file_entry["path"])
        comment_style = str(file_entry.get("comment_style") or detect_comment_style(path))
        is_cmake = str(file_entry.get("kind")) == "cmake"
        layer_id = str(file_entry.get("layer_id", ""))
        
        # Plan A: Don't create _attest.cpp file from scratch, reuse source test file instead
        is_attest_gen = layer_id in ("op_api", "op_host") and path.name.endswith("_attest.cpp")

        # Skip cmake files that don't exist yet (nothing to wrap)
        if is_cmake and not path.exists():
            return False

        # Plan A: For non-cmake operator tests, if _attest.cpp doesn't exist, copy source file
        if is_attest_gen and not path.exists() and not is_cmake:
            op_name = str(file_entry.get("op_name", ""))
            if not op_name:
                # Extract from path: .../math/<op_name>/tests/ut/op_xxx/xxx.cpp
                parts = str(path).split("/")
                for i, part in enumerate(parts):
                    if part == "math" and i + 1 < len(parts):
                        op_name = parts[i + 1]
                        break
            
            source_file = self._find_source_test_file(project_root, path, op_name)
            if source_file:
                try:
                    shutil.copy2(source_file, path)
                    bak_file = source_file.with_suffix(source_file.suffix + ".bak")
                    if source_file != path and source_file.exists() and not bak_file.exists():
                        source_file.rename(bak_file)
                    return self._ensure_skeleton(project_root, file_entry, file_cases)
                except Exception as e:
                    print(f"Warning: Failed to copy source file {source_file} to {path}: {e}")

        if path.exists():
            existing_blocks = build_block_entries(path)
            if existing_blocks:
                return False
            original = path.read_text(encoding="utf-8")
            layer_id_str = str(file_entry.get("layer_id", ""))
            if not is_cmake:
                # Non-cmake file: wrap existing content as HEADER block
                lines = [
                    start_marker("HEADER", comment_style),
                    original.rstrip(),
                    end_marker("HEADER", comment_style),
                ]
                use_stubs_existing = is_attest_gen and len(file_cases) <= _STUB_MAX_CASES_PER_FILE
                if is_attest_gen and len(file_cases) > _STUB_MAX_CASES_PER_FILE:
                    print(f"  V23: {path.name} has {len(file_cases)} cases (>{_STUB_MAX_CASES_PER_FILE}), disabling stub pre-fill")
                if use_stubs_existing:
                    for case in file_cases:
                        block_id = str(case["block_id"])
                        case_type = str(case.get("case_type") or case.get("name") or block_id)
                        lines.extend(self._case_stub_lines(block_id, case_type, comment_style))
                else:
                    for case in file_cases:
                        lines.append(placeholder_marker(str(case["block_id"]), comment_style))
                if layer_id_str == "op_host":
                    lines.extend([
                        start_marker("FOOTER", comment_style),
                        f"{comment_style} TODO: The CMake registration below should be in CMakeLists.txt FOOTER, not here",
                        f"{comment_style} if(UT_TEST_ALL OR OP_HOST_UT)",
                        f"{comment_style}     add_modules_ut_sources(UT_NAME ${{OP_INFERSHAPE_MODULE_NAME}} MODE PRIVATE DIR ${{CMAKE_CURRENT_SOURCE_DIR}})",
                        f"{comment_style}     add_modules_ut_sources(UT_NAME ${{OP_TILING_MODULE_NAME}} MODE PRIVATE DIR ${{CMAKE_CURRENT_SOURCE_DIR}})",
                        f"{comment_style} endif()",
                        end_marker("FOOTER", comment_style),
                    ])
                else:
                    lines.append(start_marker("FOOTER", comment_style))
                    lines.append(end_marker("FOOTER", comment_style))
                content = "\n".join(lines) + "\n"
                ctx = ToolContext(cwd=str(project_root), auto_approve=True)
                self.tool_runner.execute("write_file", {"path": str(file_entry["path"]), "content": content}, ctx)
                return True
            else:
                # Cmake file: wrap with HEADER + FOOTER (FOOTER needed for _attest.cpp registration)
                lines = [
                    start_marker("HEADER", comment_style),
                    original.rstrip(),
                    end_marker("HEADER", comment_style),
                ]
                if layer_id_str == "op_host":
                    lines.extend([
                        start_marker("FOOTER", comment_style),
                        f"if(UT_TEST_ALL OR OP_HOST_UT)",
                        f"    add_modules_ut_sources(UT_NAME ${{OP_INFERSHAPE_MODULE_NAME}} MODE PRIVATE DIR ${{CMAKE_CURRENT_SOURCE_DIR}})",
                        f"    add_modules_ut_sources(UT_NAME ${{OP_TILING_MODULE_NAME}} MODE PRIVATE DIR ${{CMAKE_CURRENT_SOURCE_DIR}})",
                        f"endif()",
                        end_marker("FOOTER", comment_style),
                    ])
                else:
                    lines.extend([
                        start_marker("FOOTER", comment_style),
                        end_marker("FOOTER", comment_style),
                    ])
                content = "\n".join(lines) + "\n"
                ctx = ToolContext(cwd=str(project_root), auto_approve=True)
                self.tool_runner.execute("write_file", {"path": str(file_entry["path"]), "content": content}, ctx)
                return True

        ensure_parent(path)
        op_name_b = str(file_entry.get("op_name", ""))
        if not op_name_b:
            parts = str(path).split("/")
            for i, part in enumerate(parts):
                if part == "math" and i + 1 < len(parts):
                    op_name_b = parts[i + 1]
                    break
        boilerplate = ""
        if not is_cmake and layer_id in ("op_api", "op_host") and op_name_b:
            has_source = False
            if path.parent.exists():
                source_files = [f for f in path.parent.glob("*.cpp") if not f.name.endswith("_attest.cpp")]
                if source_files:
                    has_source = True
            if not has_source:
                boilerplate = self._generate_cpp_boilerplate(layer_id, op_name_b, project_root)
        is_attest_gen_here = (
            layer_id in ("op_api", "op_host")
            and path.name.endswith("_attest.cpp")
            and not is_cmake
        )
        use_stubs = is_attest_gen_here and len(file_cases) <= _STUB_MAX_CASES_PER_FILE
        if is_attest_gen_here and len(file_cases) > _STUB_MAX_CASES_PER_FILE:
            print(f"  V23: {path.name} has {len(file_cases)} cases (>{_STUB_MAX_CASES_PER_FILE}), disabling stub pre-fill")
        lines = [
            start_marker("HEADER", comment_style),
            boilerplate.rstrip() if boilerplate else "",
            end_marker("HEADER", comment_style),
        ]
        if use_stubs:
            for case in file_cases:
                block_id = str(case["block_id"])
                case_type = str(case.get("case_type") or case.get("name") or block_id)
                lines.extend(self._case_stub_lines(block_id, case_type, comment_style))
        else:
            for case in file_cases:
                lines.append(placeholder_marker(str(case["block_id"]), comment_style))
        lines.append(start_marker("FOOTER", comment_style))
        lines.append(end_marker("FOOTER", comment_style))
        content = "\n".join(lines) + "\n"
        ctx = ToolContext(cwd=str(project_root), auto_approve=True)
        self.tool_runner.execute("write_file", {"path": str(file_entry["path"]), "content": content}, ctx)
        if not is_cmake and layer_id in ("op_api", "op_host") and op_name_b:
            cmake_path = path.parent / "CMakeLists.txt"
            if not cmake_path.exists():
                cmake_path.parent.mkdir(parents=True, exist_ok=True)
                if layer_id == "op_api":
                    cmake_content = (
                        "if(UT_TEST_ALL OR OP_API_UT)\n"
                        "    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                        "endif()\n"
                    )
                else:
                    cmake_content = (
                        "if(UT_TEST_ALL OR OP_HOST_UT)\n"
                        "    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                        "    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                        "endif()\n"
                    )
                cmake_path.write_text(cmake_content, encoding="utf-8")
        return True

    def _ensure_cmake_attest_registration(
        self,
        project_root: Path,
        file_entry: Dict[str, Any],
    ) -> None:
        file_path = str(file_entry.get("path", ""))
        if not file_path.endswith("_attest.cpp"):
            return
        layer_id = str(file_entry.get("layer_id", ""))
        cpp_name = Path(file_path).name
        cmake_path = project_root / str(Path(file_path).parent / "CMakeLists.txt")
        if not cmake_path.exists():
            cmake_path.parent.mkdir(parents=True, exist_ok=True)
            if layer_id == "op_api":
                content = (
                    "if(UT_TEST_ALL OR OP_API_UT)\n"
                    "    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                    f"    if(NOT TARGET ${{OP_API_MODULE_NAME}}_cases_obj)\n"
                    f"        add_library(${{OP_API_MODULE_NAME}}_cases_obj OBJECT)\n"
                    f"    endif()\n"
                    f"    target_sources(${{OP_API_MODULE_NAME}}_cases_obj PRIVATE {cpp_name})\n"
                    "endif()\n"
                )
            else:
                content = (
                    "if(UT_TEST_ALL OR OP_HOST_UT)\n"
                    "    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                    "    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
                    f"    if(NOT TARGET ${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj)\n"
                    f"        add_library(${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj OBJECT)\n"
                    f"    endif()\n"
                    f"    target_sources(${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj PRIVATE {cpp_name})\n"
                    "endif()\n"
                )
            cmake_path.write_text(content, encoding="utf-8")
            return
        content = cmake_path.read_text(encoding="utf-8")
        content, sanitized = _sanitize_cmake_footer(content)
        if cpp_name in content and not sanitized:
            return
        lines = content.splitlines()
        insert_idx = None
        footer_end_marker = "# ==== BLOCK:FOOTER END ===="
        for i, line in enumerate(lines):
            if footer_end_marker in line:
                insert_idx = i
                break
        if insert_idx is None:
            return
        footer_start_idx = None
        for i in range(insert_idx - 1, -1, -1):
            if "# ==== BLOCK:FOOTER START ====" in lines[i]:
                footer_start_idx = i
                break
        if footer_start_idx is not None:
            filtered = [l for l in lines[footer_start_idx + 1 : insert_idx]
                        if "add_modules_ut_sources" not in l
                        and "add_library" not in l
                        and "set(OP_API_TEST_SOURCES" not in l
                        and "set(OP_HOST_TEST_SOURCES" not in l]
            lines = lines[: footer_start_idx + 1] + filtered + lines[insert_idx:]
            insert_idx = footer_start_idx + 1 + len(filtered)
        if layer_id == "op_api":
            reg_lines = [
                f"if(NOT TARGET ${{OP_API_MODULE_NAME}}_cases_obj)",
                f"    add_library(${{OP_API_MODULE_NAME}}_cases_obj OBJECT)",
                f"endif()",
                f"target_sources(${{OP_API_MODULE_NAME}}_cases_obj PRIVATE {cpp_name})",
            ]
        else:
            reg_lines = [
                f"if(UT_TEST_ALL OR OP_HOST_UT)",
                f"    if(NOT TARGET ${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj)",
                f"        add_library(${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj OBJECT)",
                f"    endif()",
                f"    target_sources(${{OP_INFERSHAPE_MODULE_NAME}}_cases_obj PRIVATE {cpp_name})",
                f"endif()",
            ]
        new_lines = lines[:insert_idx] + reg_lines + lines[insert_idx:]
        cmake_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def _ensure_companion_cpp_files(self, project_root: Path, plan: Dict[str, Any]) -> None:
        """Create companion .cpp file alongside each *_attest.cpp.

        The CMake glob in `add_modules_ut_sources` matches *.cpp files. If the
        LLM renamed the original .cpp to .cpp.bak and no explicit target_sources
        was registered, the build falls back to compiling empty.cpp → 0 tests.
        Creating a sibling .cpp file (copy of _attest.cpp) guarantees the glob
        picks it up and compiles the test registrations — belt and suspenders on
        top of the explicit target_sources from _ensure_cmake_attest_registration.
        """
        for file_entry in _ordered_files(plan):
            if str(file_entry.get("kind", "")) == "cmake":
                continue
            path = project_root / str(file_entry.get("path", ""))
            if not path.name.endswith("_attest.cpp"):
                continue
            if not path.exists():
                continue
            # Companion = strip the `_attest` suffix: test_xxx_attest.cpp → test_xxx.cpp
            companion = path.parent / path.name.replace("_attest.cpp", ".cpp")
            if companion.exists():
                continue
            # Only create companion if the original was renamed to .bak
            # (otherwise the _attest.cpp file is already picked up by CMake glob)
            bak_path = path.parent / (path.name.replace("_attest.cpp", ".cpp") + ".bak")
            if not bak_path.exists():
                continue
            try:
                shutil.copy2(path, companion)
                print(f"  ✓ Created companion {companion.name} (copy of {path.name})")
            except Exception as exc:
                print(f"  ⚠ Failed to create companion {companion.name}: {exc}")

    def _validate_test_registrations(self, project_root: Path, plan: Dict[str, Any]) -> Dict[str, int]:
        """Count TEST_F/TEST/TEST_P macros in each generated test file.

        Returns {path: count}. Prints a warning for files with 0 registrations.
        Used to diagnose "0 tests from 0 test suites" failures early.
        """
        counts: Dict[str, int] = {}
        for file_entry in _ordered_files(plan):
            if str(file_entry.get("kind", "")) == "cmake":
                continue
            path = project_root / str(file_entry.get("path", ""))
            if not path.name.endswith(".cpp"):
                continue
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                n = (
                    len(re.findall(r"\bTEST_F\s*\(", text))
                    + len(re.findall(r"\bTEST\s*\(", text))
                    + len(re.findall(r"\bTEST_P\s*\(", text))
                )
                counts[str(path)] = n
                # Count remaining stubs separately
                stub_count = text.count("STUB: replace")
                if n == 0:
                    print(
                        f"  ⚠ {path.name}: 0 TEST_F/TEST macros found — "
                        f"this file will contribute ZERO tests; codegen likely left "
                        f"CASE blocks as empty placeholders"
                    )
                elif stub_count > 0:
                    print(
                        f"  ⚠ {path.name}: {n} TEST macro(s) registered, but "
                        f"{stub_count} STUB(s) still unfilled — coverage will be "
                        f"reduced until LLM replaces stubs with real tests"
                    )
                else:
                    print(f"  ✓ {path.name}: {n} TEST macro(s) registered, all stubs replaced")
            except Exception:
                pass
        return counts

    def _count_remaining_stubs(self, project_root: Path, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan generated .cpp files for `STUB: replace` markers that LLM didn't replace.

        Returns a list of dicts: {"file": <path>, "count": N, "block_ids": [...]}.
        Used to decide whether a focused repair session is needed.
        """
        stubs: List[Dict[str, Any]] = []
        for file_entry in _ordered_files(plan):
            if str(file_entry.get("kind", "")) == "cmake":
                continue
            path = project_root / str(file_entry.get("path", ""))
            if not path.name.endswith(".cpp") or not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            # Find each BLOCK marker + check if it contains the STUB marker comment
            block_ids_with_stub: List[str] = []
            current_block: Optional[str] = None
            for line in text.splitlines():
                m = re.match(r"^\s*(?:#|//)\s*====\s*BLOCK:(\w+)\s+START\s*====", line)
                if m:
                    current_block = m.group(1)
                    continue
                if current_block and "STUB: replace" in line:
                    block_ids_with_stub.append(current_block)
                end_m = re.match(r"^\s*(?:#|//)\s*====\s*BLOCK:\w+\s+END\s*====", line)
                if end_m:
                    current_block = None
            if block_ids_with_stub:
                stubs.append({
                    "file": str(path),
                    "count": len(block_ids_with_stub),
                    "block_ids": block_ids_with_stub,
                })
        return stubs

    def _repair_remaining_stubs(
        self,
        state,
        project_root: Path,
        plan: Dict[str, Any],
        stubs: List[Dict[str, Any]],
    ) -> None:
        """Run a focused LLM session to replace remaining STUB blocks with real tests.

        Only runs if stub count > 0. Budget: max 40 turns. The repair prompt gives the LLM
        the exact list of CASE blocks still using STUBs and asks it to visit each block with
        sed/heredoc and replace it.
        """
        if not stubs:
            return
        total = sum(s["count"] for s in stubs)
        all_block_ids: List[str] = []
        for s in stubs:
            all_block_ids.extend(s["block_ids"])
        files_desc = "\n".join(
            f"  - {s['file']} :: {s['block_ids']} ({s['count']} stubs)"
            for s in stubs
        )
        prompt = (
            f"## ⚠️ STUB REPAIR REQUIRED\n"
            f"There are {total} CASE blocks that still contain STUB markers (placeholder tests that GTEST_SKIP()). "
            f"Your job is to visit EACH file+block below with `exec_command` and REPLACE the stub with a "
            f"working `TEST_F(...)` test that actually exercises the operator code path described by `case_type`.\n\n"
            f"### Files + blocks needing repair\n{files_desc}\n\n"
            f"### Rules\n"
            f"1. For each block, `cat -n <file>` to locate it precisely, then overwrite just that block's content.\n"
            f"2. The replacement MUST use the test-fixture class defined in the HEADER (e.g. `TEST_F(XxxInferShape, ...)`).\n"
            f"3. The replacement MUST call the actual operator API under test (not `GTEST_SKIP()`).\n"
            f"4. After all repairs, run compile+test to confirm no compile errors and tests pass.\n"
            f"5. Output a final `grep -c 'STUB: replace' <file>` count = 0 for each file.\n"
            f"6. **NEVER change GTEST_SKIP() to FAIL()** — if you cannot fill a stub, leave GTEST_SKIP() in place. FAIL() in stubs crashes the entire test suite and produces 0% coverage.\n"
            f"7. Finish with JSON: ```json {{\"stubs_remaining\": 0, \"repairs_done\": {total}}} ```\n"
        )
        print(f"  🔧 Running STUB repair session for {total} remaining stubs across {len(stubs)} file(s)...")
        messages = [{"role": "user", "content": prompt}]
        append_message(
            session_id=getattr(state, "workflow_id", "workflow"),
            role="user",
            content={"stage": "generate_code", "mode": "stub_repair", "prompt": prompt},
            workspace=str(state.workspace),
            stage="generate_code",
        )
        ctx = ToolContext(cwd=str(project_root), auto_approve=True)
        tools = self._tool_schemas()
        stagnation_streak = 0
        last_sig = ""
        api429_retries = 0
        api429_max = 3
        for turn in range(40):
            resp = None
            while resp is None:
                try:
                    resp = self.llm.chat(messages, tools=tools)
                except Exception as exc:
                    exc_str = str(exc)
                    if ("429" in exc_str or "Throttl" in exc_str or "rate" in exc_str.lower()) and api429_retries < api429_max:
                        api429_retries += 1
                        delay = min(30 * api429_retries, 90)
                        print(f"  ⚠ stub repair rate limited (turn {turn+1}), retry {api429_retries}/{api429_max} after {delay}s")
                        import time
                        time.sleep(delay)
                        continue
                    print(f"  ⚠ stub repair LLM call failed turn {turn+1}: {exc}")
                    break
            if resp is None:
                break
            assistant_msg = {
                "role": "assistant",
                "content": resp.content,
                "reasoning_content": getattr(resp, "reasoning_content", "") or "",
            }
            if resp.tool_calls:
                assistant_msg["tool_calls"] = resp.tool_calls
            messages.append(assistant_msg)
            append_message(
                session_id=getattr(state, "workflow_id", "workflow"),
                role="assistant", content=assistant_msg,
                workspace=str(state.workspace), stage="generate_code",
            )
            if not resp.has_tool_calls():
                break
            for tc in resp.tool_calls:
                try:
                    tool_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    messages.append({"role": "user", "content": "Tool JSON truncated. Split into chunks."})
                    continue
                tr = self.tool_runner.execute(tc["function"]["name"], tool_args, ctx)
                out = (tr.output if tr.ok else (tr.error or ""))
                if len(out) > 6000:
                    out = out[:6000] + "\n...(truncated)"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
            sig = json.dumps([{"fn": tc["function"]["name"], "args": tc["function"]["arguments"]}
                              for tc in (resp.tool_calls or [])], sort_keys=True)
            stagnation_streak = stagnation_streak + 1 if sig == last_sig else 0
            last_sig = sig
            if stagnation_streak >= 3:
                print(f"  ⚠ stub repair stagnation at turn {turn+1}; stopping")
                break
        # Final count
        new_stubs = self._count_remaining_stubs(project_root, plan)
        if new_stubs:
            rem = sum(s["count"] for s in new_stubs)
            print(f"  ⚠ stub repair finished but {rem} stub(s) remain: {new_stubs}")
            # Safety net: replace any FAIL() in remaining stubs with GTEST_SKIP()
            self._sanitize_stub_fail_calls(project_root, plan)
        else:
            print(f"  ✓ stub repair finished: all stubs replaced")

    def _sanitize_stub_fail_calls(self, project_root: Path, plan: Dict[str, Any]) -> None:
        """Replace FAIL() with GTEST_SKIP() in any remaining STUB blocks.

        The LLM sometimes converts GTEST_SKIP() to FAIL() in stub blocks, which
        causes the entire test suite to fail and coverage to drop to 0%. This
        safety net scans all generated test files and reverts any FAIL() inside
        STUB: replace blocks back to GTEST_SKIP().
        """
        import re as _re
        for file_entry in _ordered_files(plan):
            if str(file_entry.get("kind", "")) == "cmake":
                continue
            path = project_root / str(file_entry.get("path", ""))
            if not path.name.endswith(".cpp") or not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "FAIL()" not in text or "STUB" not in text:
                continue
            # Replace FAIL() inside STUB blocks (between BLOCK:CASE_NN START/END markers
            # that contain STUB: replace)
            in_stub_block = False
            new_lines = []
            for line in text.split("\n"):
                if _re.match(r"^\s*(?:#|//)\s*====\s*BLOCK:CASE_\d+\s+START\s*====", line):
                    in_stub_block = True
                if in_stub_block and "FAIL()" in line:
                    line = line.replace("FAIL()", "GTEST_SKIP()")
                if _re.match(r"^\s*(?:#|//)\s*====\s*BLOCK:CASE_\d+\s+END\s*====", line):
                    in_stub_block = False
                new_lines.append(line)
            new_text = "\n".join(new_lines)
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                count = text.count("FAIL()") - new_text.count("FAIL()")
                print(f"  🔧 Safety net: replaced {count} FAIL() with GTEST_SKIP() in {path.name}")

    # ------------------------------------------------------------------
    # execute()
    # ------------------------------------------------------------------

    def execute(self, state) -> StageResult:
        plan = _load_json_artifact(state, "test_plan.json")
        context = _load_json_artifact(state, "operator_context.json")
        analysis_plan = _load_json_artifact(state, "analysis_plan.json", default=_default_analysis_plan())
        provider = self._skill_provider(state)
        project_root = _llm_project_root(state, context)
        generation_mode = _generation_mode(context)
        coverage_mode = _coverage_mode(context)

        if _is_enhance_mode(context):
            project_root = self._prepare_generated_repo(state, context)

        # Load cross-epoch context artifacts (P4)
        epoch = int(getattr(state, "epoch_current", 1) or 1)
        prev_error_context = ""
        case_inventory_context = ""
        if epoch > 1:
            prev_error_log = state.load_artifact("prev_epoch_error_log.txt") or ""
            if prev_error_log.strip():
                prev_error_context = (
                    f"\n## Previous Epoch Test & Coverage Results (up to 4000 chars)\n"
                    f"```\n{prev_error_log[:4000]}\n```\n"
                )
            inv = _load_json_artifact(state, "case_inventory.json", default={})
            if inv.get("generated_blocks"):
                bounded = [b["block_id"] for b in inv["generated_blocks"] if b["status"] == "bounded"]
                placeholder = [b["block_id"] for b in inv["generated_blocks"] if b["status"] == "placeholder"]
                case_inventory_context = (
                    f"\n## Previous Epoch Case Status\n"
                    f"Already bounded ({len(bounded)} total, do NOT regenerate): {', '.join(bounded[:30]) or 'none'}\n"
                    f"Still placeholder ({len(placeholder)} total, FOCUS of this epoch): {', '.join(placeholder[:30]) or 'none'}\n"
                )

        # Ensure file skeletons exist
        cases_by_file = _cases_by_file(plan)
        for file_entry in _ordered_files(plan):
            self._ensure_skeleton(project_root, file_entry, cases_by_file.get(str(file_entry["file_id"]), []))

        # Build slim_context inline (similar to standard stage)
        slim_context = {
            "op_name": context.get("op_name"),
            "generation_mode": context.get("generation_mode"),
            "coverage_mode": context.get("coverage_mode"),
            "selected_soc": context.get("selected_soc"),
            "enabled_layers": context.get("enabled_layers"),
            "is_aclnn_exclude": context.get("is_aclnn_exclude"),
            "dtype_candidates": context.get("dtype_candidates", [])[:20],
            "format_candidates": context.get("format_candidates", [])[:20],
            "build_help_summary": context.get("build_help_summary", "")[:500],
            "build_commands": context.get("build_commands", {}),
            "workspace_signatures": context.get("workspace_signatures", []),
        }
        slim_context.update(
            _read_op_source_code(
                Path(context.get("operator_dir", "")),
                str(context.get("op_name", "")),
            )
        )
        slim_context = {k: v for k, v in slim_context.items() if v is not None}

        infra_context_snippet = ""
        try:
            infra_ctx = _load_json_artifact(state, "infrastructure_context.json", default={})
            if infra_ctx:
                layout = infra_ctx.get("op_host_layout", {})
                cmake_info = infra_ctx.get("cmake_info", {})
                warnings = infra_ctx.get("warnings", [])
                infra_context_snippet_parts = ["\n## Infrastructure Context (MUST FOLLOW — verified by build probe)\n"]
                if layout.get("is_pure_opapi"):
                    infra_context_snippet_parts.append(
                        "WARNING: op_host/ contains ONLY op_api/ nesting — no infershape/tiling .cpp files. "
                        "DO NOT generate op_host tests unless you fully understand this layout. Focus on op_api tests."
                    )
                if cmake_info.get("cmake_style"):
                    infra_context_snippet_parts.append(f"CMake style: `{cmake_info['cmake_style']}`")
                if cmake_info.get("targets"):
                    infra_context_snippet_parts.append(f"CMake UT targets: `{json.dumps(cmake_info['targets'])}`")
                if cmake_info.get("test_register_macro"):
                    infra_context_snippet_parts.append(f"Test register macro: `{cmake_info['test_register_macro']}`")
                tc = cmake_info.get("test_cmake", {})
                if tc.get("op_host_op_api_exists"):
                    infra_context_snippet_parts.append(
                        "IMPORTANT: tests/ut/op_host/op_api/ exists — op_api tests should be generated "
                        "under tests/ut/op_host/op_api/, NOT under tests/ut/op_api/."
                    )
                for w in warnings:
                    infra_context_snippet_parts.append(f"⚠️ {w}")
                infra_context_snippet = "\n".join(infra_context_snippet_parts) + "\n"
                slim_context["infrastructure"] = {
                    "op_host_layout": layout,
                    "cmake_style": cmake_info.get("cmake_style"),
                    "cmake_targets": cmake_info.get("targets"),
                }
        except Exception:
            pass

        # Layers that actually have files to generate
        enabled_layers = [
            l for l in plan.get("enabled_layers", []) if self._layer_files(plan, l)
        ]
        use_parallel = self.workers > 1 and len(enabled_layers) > 1

        if not use_parallel:
            # ----- Legacy single-session path (unchanged, backward compatible) -----
            prompt = self._build_agent_loop_prompt(
                state, plan, context, project_root, slim_context, analysis_plan,
                prev_error_context, case_inventory_context,
                provider, generation_mode, coverage_mode,
                infra_context_snippet=infra_context_snippet,
            )
            stage_result = self._run_llm_session(state, prompt, project_root, turn_limit=300)

            # Post-codegen: register every *_attest.cpp in its parent CMakeLists.txt
            # FOOTER with explicit `target_sources` — prevents the glob-based
            # `add_modules_ut_sources` from falling back to empty.cpp when the
            # companion `.cpp` file is absent (Fix for diag_part-style regressions).
            for file_entry in _ordered_files(plan):
                if str(file_entry.get("kind", "")) != "cmake":
                    self._ensure_cmake_attest_registration(project_root, file_entry)

            # Post-codegen: create companion .cpp files so CMake glob picks up tests
            # even if the original .cpp was renamed to .bak (belt and suspenders).
            self._ensure_companion_cpp_files(project_root, plan)

            # Post-codegen: count TEST_F macros to catch empty-CASE regressions early
            self._validate_test_registrations(project_root, plan)

            # Post-codegen: run focused repair session for any CASE blocks LLM left
            # as STUBs (the TEST(AttestStubs, ...) placeholder) so coverage improves.
            remaining = self._count_remaining_stubs(project_root, plan)
            if remaining:
                self._repair_remaining_stubs(state, project_root, plan, remaining)
                # Re-count after repair
                self._validate_test_registrations(project_root, plan)
            # Safety net: ensure no FAIL() in stub blocks
            self._sanitize_stub_fail_calls(project_root, plan)

            manifest = [
                {
                    "file_id": str(f.get("file_id", "")),
                    "path": f.get("path", ""),
                    "action": "updated",
                    "target_blocks": ["ALL"],
                }
                for f in plan.get("files", [])
                if isinstance(f, dict)
            ]
            manifest_text = _render_json({"files": manifest, "project_root": str(project_root)})
            state.save_artifact("generation_manifest.json", manifest_text)

            exec_stage = AscendExecutionStage(self.llm, self.tool_runner)
            exec_result = exec_stage.execute(state)
            gen_success = stage_result.success
            exec_outputs = exec_result.outputs or {}
            exec_ok = exec_result.success
        else:
            # ----- Parallel per-layer generation path -----
            print(f"\n  ⚡ Parallel generation: {len(enabled_layers)} layer agents "
                  f"({', '.join(enabled_layers)}), workers={self.workers}")
            build_sh_path = project_root / "build.sh"
            build_sh_backup: Optional[Path] = None
            build_sh_patched = False
            # Patch build.sh ONCE up-front on the main thread; worker threads never
            # touch it. This eliminates the patch/restore race under parallelism.
            if build_sh_path.exists():
                build_sh_backup = build_sh_path.with_suffix(".sh.parallel.bak")
                shutil.copy2(build_sh_path, build_sh_backup)
                _patch_build_sh_for_isolation(project_root)
                build_sh_patched = True
                _ensure_cann_cmake_available(project_root)

            for file_entry in _ordered_files(plan):
                if str(file_entry.get("kind", "")) != "cmake":
                    self._ensure_cmake_attest_registration(project_root, file_entry)

            gen_success = False
            layer_summaries: Dict[str, Any] = {}
            try:
                with ThreadPoolExecutor(max_workers=min(self.workers, len(enabled_layers))) as pool:
                    futures = {
                        pool.submit(
                            self._run_layer_session,
                            state, plan, layer, project_root, slim_context,
                            analysis_plan, prev_error_context, case_inventory_context,
                            provider, generation_mode,
                            infra_context_snippet=infra_context_snippet,
                        ): layer
                        for layer in enabled_layers
                    }
                    for fut in as_completed(futures):
                        layer = futures[fut]
                        try:
                            _layer, res, summ = fut.result()
                            layer_summaries[_layer] = summ
                            if res.success:
                                gen_success = True
                            print(f"  ✓ layer `{_layer}` generation session done "
                                  f"(success={res.success})")
                        except Exception as exc:
                            print(f"  ✗ layer `{layer}` generation raised: {exc}")

                # CMake registration safety net (main thread) — ensures every
                # *_attest.cpp is registered in its parent CMakeLists.txt FOOTER,
                # even if the LLM agent forgot or wrote incorrect registration.
                for file_entry in _ordered_files(plan):
                    if str(file_entry.get("kind", "")) != "cmake":
                        self._ensure_cmake_attest_registration(project_root, file_entry)

                # Companion .cpp file creation — ensures CMake glob also finds tests
                self._ensure_companion_cpp_files(project_root, plan)

                # Count TEST_F macros to catch empty-CASE regressions early
                self._validate_test_registrations(project_root, plan)

                # Run focused repair for any CASE blocks still using STUBs
                remaining = self._count_remaining_stubs(project_root, plan)
                if remaining:
                    self._repair_remaining_stubs(state, project_root, plan, remaining)
                    self._validate_test_registrations(project_root, plan)
                # Safety net: ensure no FAIL() in stub blocks
                self._sanitize_stub_fail_calls(project_root, plan)

                # Manifest (main thread)
                manifest = [
                    {
                        "file_id": str(f.get("file_id", "")),
                        "path": f.get("path", ""),
                        "action": "updated",
                        "target_blocks": ["ALL"],
                    }
                    for f in plan.get("files", [])
                    if isinstance(f, dict)
                ]
                manifest_text = _render_json({"files": manifest, "project_root": str(project_root)})
                state.save_artifact("generation_manifest.json", manifest_text)

                # Per-layer coverage builds run SEQUENTIALLY on the main thread
                cov = self._collect_parallel_coverage(state, plan, project_root, enabled_layers)
                exec_outputs = {
                    "execution_log.txt": cov["execution_log.txt"],
                    "exit_code.txt": cov["exit_code.txt"],
                    "coverage_summary.json": cov["coverage_summary.json"],
                }
                exec_ok = bool((cov.get("summary") or {}).get("coverage_valid", False))
            finally:
                # Always restore build.sh (crash-safe), mirroring _run_for_root finally
                if build_sh_patched and build_sh_backup and build_sh_backup.exists():
                    try:
                        shutil.copy2(build_sh_backup, build_sh_path)
                        build_sh_backup.unlink()
                    except Exception as exc:
                        print(f"  ⚠️  failed to restore build.sh: {exc}")

        # ------------------------------------------------------------------
        # Run analysis (reuse AnalysisStage logic) — same for both paths
        # ------------------------------------------------------------------
        analysis_stage = AscendAnalysisStage(self.llm, self.tool_runner)
        analysis_result = analysis_stage.execute(state)

        # ------------------------------------------------------------------
        # Return combined result
        # ------------------------------------------------------------------
        success = gen_success or exec_ok
        combined_outputs: Dict[str, Any] = {}
        combined_outputs["generation_manifest.json"] = manifest_text
        combined_outputs.update(exec_outputs or {})
        combined_outputs.update(analysis_result.outputs or {})

        cov_summary = _load_json_artifact(state, "coverage_summary.json", default={})
        cov_note = ""
        for layer in plan.get("enabled_layers", []):
            ldata = cov_summary.get("per_layer", {}).get(str(layer), {})
            lc = ldata.get("line_coverage")
            if lc is not None:
                cov_note += f" {layer}={lc:.1f}%"

        return StageResult(
            success,
            combined_outputs,
            message=f"Continuous agent loop complete.{cov_note}",
        )


class AscendReportStage(AscendBaseStage):
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="generate_report",
            display_name="Generate Report",
            description="Summarize Ascend UT generation outputs",
            prompt_template="",
            input_artifacts=["operator_context.json", "requirements.md", "test_plan.json", "analysis.md", "coverage_summary.json"],
            output_artifacts=["final_report.md"],
            tools=[],
            allow_skip=False,
        )

    def execute(self, state) -> StageResult:
        context = _load_json_artifact(state, "operator_context.json")
        plan = _load_json_artifact(state, "test_plan.json")
        coverage = _load_json_artifact(state, "coverage_summary.json", default={})
        analysis_md = str(state.load_artifact("analysis.md") or "")
        provider = self._skill_provider(state)

        lines = [
            f"# Final Report - {context.get('op_path', state.target)}",
            "",
            "## Summary",
            f"- Generation mode: `{context.get('generation_mode', 'unknown')}`",
            f"- Coverage mode: `{plan.get('coverage_mode', context.get('coverage_mode', 'unknown'))}`",
            f"- Enabled layers: {', '.join(context.get('enabled_layers', [])) or 'none'}",
            f"- Selected SoC: `{context.get('selected_soc', state.soc)}`",
            f"- Auto-stop reason: `{getattr(state, 'auto_stop_reason', '') or 'none'}`",
            "",
            "## Generated Files",
        ]
        for entry in _ordered_files(plan):
            lines.append(f"- `{entry['path']}` ({entry['layer_id']}, {entry['kind']})")

        lines.extend(["", "## Coverage"])
        if coverage.get("mode") == "before_after_compare":
            lines.append(f"- Baseline root: `{coverage.get('baseline_root', plan.get('baseline', {}).get('project_root', ''))}`")
            lines.append(f"- Generated root: `{coverage.get('generated_root', plan.get('generated', {}).get('project_root', ''))}`")
            for layer, meta in (coverage.get("per_layer") or {}).items():
                baseline_meta = (meta.get("baseline") or {}) if isinstance(meta, dict) else {}
                generated_meta = (meta.get("generated") or {}) if isinstance(meta, dict) else {}
                baseline_cov = float(baseline_meta.get("line_coverage", 0.0))
                generated_cov = float(generated_meta.get("line_coverage", 0.0))
                delta = float(meta.get("delta_line_coverage", 0.0)) if isinstance(meta, dict) else 0.0
                branch_text = ""
                if baseline_meta.get("branch_coverage") is not None or generated_meta.get("branch_coverage") is not None:
                    baseline_branch = baseline_meta.get("branch_coverage")
                    generated_branch = generated_meta.get("branch_coverage")
                    delta_branch = meta.get("delta_branch_coverage") if isinstance(meta, dict) else None
                    branch_text = (
                        f"; branch {baseline_branch}% -> {generated_branch}%"
                        f" (delta {float(delta_branch):+.2f}%)"
                        if delta_branch is not None
                        else f"; branch {baseline_branch}% -> {generated_branch}%"
                    )
                lines.append(
                    f"- `{layer}`: line baseline {baseline_cov}% -> generated {generated_cov}% "
                    f"(delta {delta:+.2f}%){branch_text}"
                )
            overall = coverage.get("overall", {})
            lines.append(
                f"- Overall generated>=baseline: `{overall.get('generated_ge_baseline', False)}`; "
                f"generated meets threshold: `{overall.get('generated_meets_threshold', False)}`"
            )
        else:
            for layer, meta in (coverage.get("per_layer") or {}).items():
                branch_text = (
                    f", branch {meta.get('branch_coverage')}%"
                    if meta.get("branch_coverage") is not None
                    else ""
                )
                lines.append(
                    f"- `{layer}`: line {meta.get('line_coverage', 0.0)}%{branch_text} "
                    f"(meets threshold: {meta.get('meets_threshold', False)})"
                )

        lines.extend(
            [
                "",
                "## Analysis Snapshot",
                analysis_md,
                "",
                "## Skill Summary",
                "```text",
                provider.get_stage_packet(
                    "generate_report",
                    generation_mode=plan.get("generation_mode", context.get("generation_mode", "ut_generate")),
                ),
                "```",
            ]
        )

        content = "\n".join(lines)
        state.save_artifact("final_report.md", content)
        return StageResult(True, {"final_report.md": content}, message="Final Ascend UT report generated")


class AscendValidateInfrastructureStage(AscendBaseStage):
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="validate_infrastructure",
            display_name="Validate Infrastructure",
            description="Probe operator build system and coverage pipeline to pre-resolve infrastructure issues",
            prompt_template="",
            input_artifacts=["operator_context.json"],
            output_artifacts=["infrastructure_context.json", "infrastructure_report.md"],
            tools=["exec_command", "read_file"],
            allow_skip=False,
        )

    def _run_smoke_probe(
        self,
        state,
        op_dir: Path,
        infra_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        ctx = ToolContext(cwd=str(state.project_root), auto_approve=True)
        build_cmd_template = infra_ctx.get("build_cmd_template", "")
        if not build_cmd_template:
            return {"smoke_build_ran": False, "reason": "no build command template available"}
        op_name = state.op_name
        build_dir = f"build_infra_probe_{op_name}"
        build_out_dir = f"build_out_infra_probe_{op_name}"
        project_root = state.project_root
        build_sh_path = project_root / "build.sh"
        build_sh_backup: Optional[Path] = None
        if build_sh_path.exists():
            build_sh_backup = build_sh_path.with_suffix(".sh.probe.bak")
            shutil.copy2(build_sh_path, build_sh_backup)
        _patch_build_sh_for_isolation(project_root)
        _ensure_cann_cmake_available(project_root)
        env_prefix = (
            f"BUILD_PATH={shlex.quote(str(project_root / build_dir))} "
            f"BUILD_OUT_PATH={shlex.quote(str(project_root / build_out_dir))} "
        )
        probe_cmd = f"{env_prefix}{build_cmd_template}"
        result = self.tool_runner.execute("exec_command", {"cmd": probe_cmd}, ctx, source="framework")
        output = result.output if result.output else result.error or ""
        gcda_cmd = f"find {shlex.quote(str(project_root / build_dir))} -name '*.gcda' 2>/dev/null | wc -l"
        gcda_result = self.tool_runner.execute("exec_command", {"cmd": gcda_cmd}, ctx, source="framework")
        gcda_count = 0
        try:
            gcda_count = int((gcda_result.output or "0").strip())
        except (ValueError, TypeError):
            pass
        build_dir_path = project_root / build_dir
        coverage_ok = gcda_count > 7
        warnings: List[str] = []
        if "Running 0 tests" in output:
            warnings.append("Smoke build ran 0 tests — test binary has no TEST_F registrations")
        if gcda_count <= 7:
            warnings.append(
                f"Smoke build produced only {gcda_count} .gcda files (≤7 = test framework only, "
                "no operator code executed)"
            )
        cleanup_cmd = f"rm -rf {shlex.quote(str(project_root / build_dir))} {shlex.quote(str(project_root / build_out_dir))}"
        self.tool_runner.execute("exec_command", {"cmd": cleanup_cmd}, ctx, source="framework")
        if build_sh_backup and build_sh_backup.exists():
            try:
                shutil.copy2(build_sh_backup, build_sh_path)
                build_sh_backup.unlink()
            except Exception:
                pass
        return {
            "smoke_build_ran": True,
            "smoke_build_ok": bool(result.ok),
            "gcda_count": gcda_count,
            "coverage_ok": coverage_ok,
            "warnings": warnings,
            "build_log_snippet": output[:1500] if not coverage_ok else "",
        }

    def execute(self, state) -> StageResult:
        context = _load_json_artifact(state, "operator_context.json")
        if not context:
            return StageResult(False, {}, error="operator_context.json not available")
        op_dir_str = context.get("operator_dir", "")
        if not op_dir_str:
            return StageResult(False, {}, error="operator_dir not found in operator_context")
        op_dir = Path(op_dir_str)
        if not op_dir.is_dir():
            return StageResult(False, {}, error=f"operator_dir does not exist: {op_dir}")
        infra_ctx = _collect_infrastructure_context(state, op_dir, context)
        smoke = self._run_smoke_probe(state, op_dir, infra_ctx)
        infra_ctx["smoke_probe"] = smoke
        for w in smoke.get("warnings", []):
            infra_ctx.setdefault("warnings", []).append(w)
        json_content = _render_json(infra_ctx)
        state.save_artifact("infrastructure_context.json", json_content)
        md_lines = [
            f"# Infrastructure Validation Report — {state.op_name}",
            "",
            "## Directory Layout",
            f"- op_host layout: `{'pure_opapi' if infra_ctx['op_host_layout'].get('is_pure_opapi') else 'standard'}`",
            f"- op_host has real host code: {infra_ctx['op_host_layout'].get('has_real_host_code', False)}",
            f"- op_host has op_api subdir: {infra_ctx['op_host_layout'].get('has_op_api_subdir', False)}",
            "",
            "## CMake",
            f"- Style: `{infra_ctx['cmake_info'].get('cmake_style', 'unknown')}`",
            f"- UT CMake targets: {json.dumps(infra_ctx['cmake_info'].get('targets', {}))}",
            f"- Test register macro: `{infra_ctx['cmake_info'].get('test_register_macro', 'add_modules_ut_sources')}`",
            f"- test_cmake: {json.dumps(infra_ctx['cmake_info'].get('test_cmake', {}))}",
            "",
            "## Smoke Probe",
            f"- Build OK: {smoke.get('smoke_build_ok', False)}",
            f"- .gcda files: {smoke.get('gcda_count', 0)}",
            f"- Coverage OK: {smoke.get('coverage_ok', False)}",
        ]
        if smoke.get("warnings"):
            md_lines.extend(["", "## Smoke Warnings"])
            md_lines.extend(f"- {w}" for w in smoke["warnings"])
        if infra_ctx.get("warnings"):
            md_lines.extend(["", "## All Warnings"])
            seen = set()
            for w in infra_ctx["warnings"]:
                if w not in seen:
                    md_lines.append(f"- {w}")
                    seen.add(w)
        if infra_ctx["op_host_layout"].get("is_pure_opapi"):
            md_lines.extend([
                "",
                "## LCOV Override",
                "op_host `--remove` pattern is **disabled** because op_host/ contains only op_api/ — removing would wipe all records.",
            ])
        markdown = "\n".join(md_lines)
        state.save_artifact("infrastructure_report.md", markdown)
        outputs = {
            "infrastructure_context.json": json_content,
            "infrastructure_report.md": markdown,
        }
        msg = f"Infrastructure validated for '{state.op_name}'"
        if infra_ctx.get("warnings"):
            msg += f" ({len(infra_ctx['warnings'])} warning(s))"
        return StageResult(True, outputs, message=msg)


def build_ascend_stages(llm, tool_runner):
    return {
        "understand_function": AscendInspectOperatorStage(llm, tool_runner),
        "generate_requirements": AscendRequirementsStage(llm, tool_runner),
        "design_test_plan": AscendTestPlanStage(llm, tool_runner),
        "validate_infrastructure": AscendValidateInfrastructureStage(llm, tool_runner),
        "generate_code": AscendCodeGenStage(llm, tool_runner),
        "execute_tests": AscendExecutionStage(llm, tool_runner),
        "analyze_results": AscendAnalysisStage(llm, tool_runner),
        "generate_report": AscendReportStage(llm, tool_runner),
    }


def build_ascend_continuous_stages(llm, tool_runner):
    """Stage map for the `ascend_ut_continuous` profile.

    Stages 4+5+6 are replaced by a single AscendGenerationAgentLoopStage that
    runs in one continuous LLM session. Stage 7 (generate_report) is unchanged.
    Stage 3b (validate_infrastructure) probes the build system before code generation.
    """
    return {
        "understand_function": AscendInspectOperatorStage(llm, tool_runner),
        "generate_requirements": AscendRequirementsStage(llm, tool_runner),
        "design_test_plan": AscendTestPlanStage(llm, tool_runner),
        "validate_infrastructure": AscendValidateInfrastructureStage(llm, tool_runner),
        "generate_code": AscendGenerationAgentLoopStage(llm, tool_runner),
        "generate_report": AscendReportStage(llm, tool_runner),
    }
