"""
Deterministic repository inspection helpers for Ascend operator UT generation.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SOC_RE = re.compile(r"Ascend[0-9A-Za-z]+")
DTYPE_RE = re.compile(r"DT_[A-Z0-9_]+")
FORMAT_RE = re.compile(r"FORMAT_[A-Z0-9_]+")
ATTR_RE = re.compile(r'GetAttr(?:Value)?(?:<[^>]+>)?\s*\(\s*"([^"]+)"')
TILING_PARSE_RE = re.compile(r"TilingParse<\s*([A-Za-z0-9_:]+)\s*>")
COMPARE_LAYERS = ("op_host", "op_api")


def normalize_operator_path(project_root: str | Path, op_path: str) -> Dict[str, str]:
    root = Path(project_root).resolve()
    parts = [part for part in Path(op_path).parts if part not in {".", ""}]
    if len(parts) < 2:
        raise ValueError("op_path must be '<category>/<op>' or '<repo>/<category>/<op>'")

    repo_name = root.name
    warnings: List[str] = []

    if len(parts) == 2:
        category, op_name = parts
        canonical_repo = repo_name
    else:
        candidate_repo, category, op_name = parts[-3], parts[-2], parts[-1]
        canonical_repo = candidate_repo or repo_name
        if candidate_repo and candidate_repo != repo_name:
            warnings.append(
                f"Provided repo name '{candidate_repo}' does not match project root '{repo_name}'; using project root"
            )
            canonical_repo = repo_name

    return {
        "repo_name": canonical_repo,
        "category": category,
        "op_name": op_name,
        "canonical_op_path": f"{canonical_repo}/{category}/{op_name}",
        "relative_op_dir": f"{category}/{op_name}",
        "warnings": warnings,
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _find_unique(pattern: re.Pattern[str], files: Iterable[Path]) -> List[str]:
    values = set()
    for file_path in files:
        values.update(pattern.findall(_read_text(file_path)))
    return sorted(values)


def _run_help(project_root: Path) -> str:
    build_script = project_root / "build.sh"
    if not build_script.exists():
        return ""
    try:
        result = subprocess.run(
            ["bash", "build.sh", "-h"],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return ""
    return "\n".join(
        line for line in (result.stdout + "\n" + result.stderr).splitlines() if line.strip()
    )


def _summarize_help_output(help_text: str) -> str:
    if not help_text.strip():
        return "build.sh -h unavailable"
    selected: List[str] = []
    for line in help_text.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in ("--ophost", "--opapi", "--opkernel", "--soc", "--cov")):
            selected.append(line.strip())
    if not selected:
        selected = [line.strip() for line in help_text.splitlines()[:12] if line.strip()]
    return "\n".join(selected[:20])


def _extract_supported_socs(project_root: Path, help_text: str) -> List[str]:
    values = set(SOC_RE.findall(help_text))
    build_script = project_root / "build.sh"
    if build_script.exists():
        values.update(SOC_RE.findall(_read_text(build_script)))
    return sorted(values)


def _detect_aclnn_exclude(op_dir: Path) -> bool:
    for cmake_path in sorted(op_dir.rglob("CMakeLists.txt")):
        content = _read_text(cmake_path)
        if re.search(r"ACLNNTYPE\s+aclnn_exclude", content, re.IGNORECASE):
            return True
    return False


def _detect_layers(op_dir: Path) -> Dict[str, Dict[str, Any]]:
    layers: Dict[str, Dict[str, Any]] = {}

    op_host_path = op_dir / "op_host"
    op_api_candidates = [
        op_dir / "op_api",
        op_dir / "op_host" / "op_api",
    ]
    op_api_path = next((path for path in op_api_candidates if path.is_dir()), op_dir / "op_api")

    layer_paths = {
        "op_host": op_host_path,
        "op_api": op_api_path,
        "op_kernel": op_dir / "op_kernel",
        "op_kernel_aicpu": op_dir / "op_kernel_aicpu",
    }

    for layer_name, layer_path in layer_paths.items():
        layers[layer_name] = {
            "exists": layer_path.is_dir(),
            "path": str(layer_path),
            "required": layer_name == "op_host",
            "comparable": layer_name in COMPARE_LAYERS,
        }
    return layers


def _detect_existing_ut(op_dir: Path, layers: Dict[str, Dict[str, Any]]) -> Tuple[bool, List[str], Dict[str, List[str]]]:
    ut_root = op_dir / "tests" / "ut"
    if not ut_root.exists():
        return False, [], {layer: [] for layer in COMPARE_LAYERS}

    by_layer: Dict[str, List[str]] = {layer: [] for layer in COMPARE_LAYERS}
    for layer_name in COMPARE_LAYERS:
        if not layers.get(layer_name, {}).get("exists"):
            continue
        layer_dir = ut_root / layer_name
        if not layer_dir.exists():
            continue
        matches = sorted(str(path.relative_to(op_dir)) for path in layer_dir.rglob("test_*.cpp"))
        by_layer[layer_name] = matches[:20]

    combined = []
    for layer_name in COMPARE_LAYERS:
        combined.extend(by_layer.get(layer_name, []))
    combined = sorted(set(combined))
    return bool(combined), combined[:20], by_layer


def _compile_info_candidates(op_host_dir: Path) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    if not op_host_dir.exists():
        return candidates

    for source_path in sorted(op_host_dir.rglob("*.cpp")):
        text = _read_text(source_path)
        for raw_type in TILING_PARSE_RE.findall(text):
            namespace = ""
            type_name = raw_type
            if "::" in raw_type:
                namespace, type_name = raw_type.rsplit("::", 1)
            elif raw_type == "BroadcastCompileInfo":
                namespace = "Ops::Base"
            elif raw_type.endswith("CompileInfo"):
                namespace = "optiling"

            header = ""
            if type_name == "BroadcastCompileInfo":
                header = "atvoss/broadcast/broadcast_tiling.h"
            elif type_name.endswith("CompileInfo"):
                header = source_path.name.replace(".cpp", ".h")

            candidate = {
                "type": type_name,
                "namespace": namespace,
                "header": header,
                "source_file": str(source_path),
            }
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates[:10]


def _required_attrs(op_host_dir: Path) -> List[str]:
    if not op_host_dir.exists():
        return []
    values = set()
    for path in sorted(op_host_dir.rglob("*.cpp")):
        values.update(ATTR_RE.findall(_read_text(path)))
    return sorted(values)


def _reference_ut_paths(project_root: Path, category: str, op_name: str) -> List[str]:
    base = project_root / category
    if not base.exists():
        return []
    matches: List[str] = []
    for path in sorted(base.rglob("tests/ut/**/test_*.cpp")):
        if f"/{op_name}/" in str(path).replace("\\", "/"):
            continue
        matches.append(str(path.relative_to(project_root)))
        if len(matches) >= 12:
            break
    return matches


def _shared_test_harness_paths(project_root: Path) -> Dict[str, List[str]]:
    shared_test_harness: List[str] = []
    shared_helpers: List[str] = []

    for candidate in [
        project_root / "tests" / "ut" / "op_host" / "test_op_host_main.cpp",
        project_root / "tests" / "ut" / "op_api" / "test_op_api_main.cpp",
        project_root / "tests" / "ut" / "common" / "CMakeLists.txt",
        project_root / "tests" / "ut" / "op_host" / "CMakeLists.txt",
        project_root / "tests" / "ut" / "op_api" / "CMakeLists.txt",
        project_root / "tests" / "ut" / "common" / "infershape_context_faker.h",
        project_root / "tests" / "ut" / "common" / "infershape_context_faker.cpp",
        project_root / "tests" / "ut" / "common" / "tiling_context_faker.h",
        project_root / "tests" / "ut" / "common" / "tiling_context_faker.cpp",
        project_root / "tests" / "ut" / "common" / "infershape_case_executor.cpp",
        project_root / "tests" / "ut" / "common" / "tiling_case_executor.cpp",
        project_root / "tests" / "ut" / "common" / "any_value.h",
    ]:
        if candidate.exists():
            shared_test_harness.append(str(candidate.relative_to(project_root)))

    for candidate in [
        project_root / "common" / "inc" / "op_host" / "tiling_base.h",
        project_root / "common" / "inc" / "op_host" / "tiling_templates_registry.h",
        project_root / "common" / "inc" / "op_host" / "tiling_util.h",
        project_root / "common" / "inc" / "op_host" / "input_util.h",
    ]:
        if candidate.exists():
            shared_helpers.append(str(candidate.relative_to(project_root)))

    return {
        "shared_test_harness": shared_test_harness[:12],
        "shared_helpers": shared_helpers[:12],
    }


def _operator_impl_files(project_root: Path, op_dir: Path) -> List[str]:
    impl_files: List[str] = []
    for base in [op_dir / "op_host", op_dir / "op_api"]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.cpp")):
            impl_files.append(str(path.relative_to(project_root)))
            if len(impl_files) >= 20:
                return impl_files
    return impl_files


_WS_SIG_RE = re.compile(
    r"aclnn(\w+)GetWorkspaceSize\s*\(",
)


def _extract_workspace_signatures(op_dir: Path) -> List[Dict[str, Any]]:
    op_api_dir = op_dir / "op_api"
    if not op_api_dir.exists():
        return []
    header_files = sorted(op_api_dir.glob("*.h"))
    results: List[Dict[str, Any]] = []
    for header_path in header_files:
        content = _read_text(header_path)
        if not content.strip():
            continue
        clean = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        clean = re.sub(r"//.*$", "", clean, flags=re.MULTILINE)
        for match in _WS_SIG_RE.finditer(clean):
            func_base_name = f"aclnn{match.group(1)}GetWorkspaceSize"
            pos = match.end()
            depth = 1
            end = pos
            while end < len(clean) and depth > 0:
                if clean[end] == "(":
                    depth += 1
                elif clean[end] == ")":
                    depth -= 1
                end += 1
            if depth != 0:
                continue
            params_raw = clean[pos:end - 1].strip()
            params_clean = re.sub(r"\s+", " ", params_raw).strip()
            params: List[Dict[str, str]] = []
            if params_clean:
                for p in params_clean.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    is_const = p.strip().startswith("const")
                    parts = p.strip().split()
                    p_name = parts[-1].strip("*") if parts else ""
                    p_type = " ".join(parts[:-1]) if len(parts) > 1 else p
                    if const_match := re.match(r"^(const\s+)?(.+?)\s*\*?\s*(\w+)$", p.strip()):
                        p_type = (const_match.group(1) or "") + const_match.group(2)
                        p_name = const_match.group(3)
                    kind = "skip"
                    if p_name in {"workspaceSize", "executor"}:
                        kind = "skip"
                    elif is_const:
                        kind = "input"
                    elif p_name.endswith("Ref"):
                        kind = "input"
                    elif p_name in {"modifiedInPlace", "mode", "axis", "dim", "keepDim", "sorted", "descending"}:
                        kind = "input"
                    else:
                        kind = "output"
                    params.append({"name": p_name, "type": p_type.strip(), "kind": kind})
            inputs = [p["name"] for p in params if p["kind"] == "input"]
            outputs = [p["name"] for p in params if p["kind"] == "output"]
            if not inputs and not outputs:
                continue
            results.append({
                "function": func_base_name,
                "signature": f"{func_base_name}({params_clean})",
                "input_params": inputs,
                "output_params": outputs,
            })
    return results


def _companion_path(existing_paths: List[str], default_path: str, stem_token: str, suffix_name: str, op_prefix: str) -> str:
    for raw_path in existing_paths:
        candidate = Path(raw_path)
        if stem_token not in candidate.name:
            continue
        return str(Path(op_prefix) / candidate.with_name(suffix_name))
    return default_path


def _suggested_files(
    category: str,
    op_name: str,
    enabled_layers: List[str],
    generation_mode: str,
    existing_ut_by_layer: Optional[Dict[str, List[str]]] = None,
    is_aclnn_exclude: bool = False,
) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    next_index = 1
    existing_ut_by_layer = existing_ut_by_layer or {}
    op_prefix = f"{category}/{op_name}"

    def add(layer_id: str, kind: str, relative_path: str, comment_style: str) -> None:
        nonlocal next_index
        file_id = f"FILE_{next_index:02d}_{layer_id.upper()}_{kind.upper()}"
        next_index += 1
        files.append(
            {
                "file_id": file_id,
                "layer_id": layer_id,
                "kind": kind,
                "path": relative_path,
                "comment_style": comment_style,
            }
        )

    if generation_mode == "ut_generate":
        if "op_host" in enabled_layers:
            add("op_host", "cmake", f"{category}/{op_name}/tests/ut/op_host/CMakeLists.txt", "#")
            if not is_aclnn_exclude:
                add("op_host", "cpp", f"{category}/{op_name}/tests/ut/op_host/test_{op_name}_tiling.cpp", "//")
            add("op_host", "cpp", f"{category}/{op_name}/tests/ut/op_host/test_{op_name}_infershape.cpp", "//")
        if "op_api" in enabled_layers:
            add("op_api", "cmake", f"{category}/{op_name}/tests/ut/op_api/CMakeLists.txt", "#")
            add("op_api", "cpp", f"{category}/{op_name}/tests/ut/op_api/test_aclnn_{op_name}.cpp", "//")
        return files

    if "op_host" in enabled_layers:
        op_host_existing = existing_ut_by_layer.get("op_host", [])
        add("op_host", "cmake", f"{category}/{op_name}/tests/ut/op_host/CMakeLists.txt", "#")
        if not is_aclnn_exclude:
            has_tiling = any("tiling" in Path(p).name for p in op_host_existing)
            tiling_name = (
                f"test_{op_name}_tiling.cpp" if not has_tiling else f"test_{op_name}_tiling_attest.cpp"
            )
            add(
                "op_host",
                "cpp",
                _companion_path(
                    op_host_existing,
                    f"{category}/{op_name}/tests/ut/op_host/{tiling_name}",
                    "tiling",
                    tiling_name,
                    op_prefix,
                ),
                "//",
            )
        has_infershape = any("infershape" in Path(p).name for p in op_host_existing)
        infershape_name = (
            f"test_{op_name}_infershape.cpp" if not has_infershape else f"test_{op_name}_infershape_attest.cpp"
        )
        add(
            "op_host",
            "cpp",
            _companion_path(
                op_host_existing,
                f"{category}/{op_name}/tests/ut/op_host/{infershape_name}",
                "infershape",
                infershape_name,
                op_prefix,
            ),
            "//",
        )
    if "op_api" in enabled_layers:
        op_api_existing = existing_ut_by_layer.get("op_api", [])
        has_op_api = bool(op_api_existing)
        add("op_api", "cmake", f"{category}/{op_name}/tests/ut/op_api/CMakeLists.txt", "#")
        api_name = (
            f"test_aclnn_{op_name}_attest.cpp" if has_op_api else f"test_aclnn_{op_name}.cpp"
        )
        add(
            "op_api",
            "cpp",
            _companion_path(
                op_api_existing,
                f"{category}/{op_name}/tests/ut/op_api/{api_name}",
                "aclnn",
                api_name,
                op_prefix,
            ),
            "//",
        )
    return files


def inspect_ascend_operator(
    project_root: str | Path,
    op_path: str,
    soc_hint: str = "",
) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    normalized = normalize_operator_path(root, op_path)
    op_dir = root / normalized["category"] / normalized["op_name"]
    help_text = _run_help(root)
    supported_socs = _extract_supported_socs(root, help_text)
    selected_soc = soc_hint or (supported_socs[0] if supported_socs else "Ascend910B")

    layers = _detect_layers(op_dir)
    enabled_layers = [name for name, meta in layers.items() if meta["exists"] and meta.get("comparable")]
    existing_ut, existing_ut_files, existing_ut_by_layer = _detect_existing_ut(op_dir, layers)
    generation_mode = "ut_enhance" if existing_ut else "ut_generate"
    _force_cov = os.environ.get("ATTEST_FORCE_COVERAGE_MODE")
    coverage_mode = (
        _force_cov if _force_cov in ("single_run", "before_after_compare")
        else ("before_after_compare" if existing_ut else "single_run")
    )
    excluded_layers = [name for name, meta in layers.items() if meta["exists"] and not meta.get("comparable")]
    is_aclnn_exclude = _detect_aclnn_exclude(op_dir)

    op_host_dir = op_dir / "op_host"
    source_files = sorted(op_host_dir.rglob("*.cpp")) if op_host_dir.exists() else []
    dtype_candidates = _find_unique(DTYPE_RE, source_files)
    format_candidates = _find_unique(FORMAT_RE, source_files)

    result: Dict[str, Any] = {
        "workflow_kind": "ascend_ut",
        "generation_mode": generation_mode,
        "coverage_mode": coverage_mode,
        "existing_ut_detected": existing_ut,
        "existing_ut_files": existing_ut_files,
        "existing_ut_by_layer": existing_ut_by_layer,
        "is_aclnn_exclude": is_aclnn_exclude,
        "repo_name": normalized["repo_name"],
        "category": normalized["category"],
        "op_name": normalized["op_name"],
        "op_path": normalized["canonical_op_path"],
        "relative_op_dir": normalized["relative_op_dir"],
        "project_root": str(root),
        "operator_dir": str(op_dir),
        "selected_soc": selected_soc,
        "supported_socs": supported_socs,
        "layers": layers,
        "enabled_layers": enabled_layers,
        "compare_scope": list(COMPARE_LAYERS),
        "excluded_layers": excluded_layers,
        "tests_root": str(op_dir / "tests" / "ut"),
        "suggested_files": _suggested_files(
            normalized["category"],
            normalized["op_name"],
            enabled_layers,
            generation_mode,
            existing_ut_by_layer=existing_ut_by_layer,
            is_aclnn_exclude=is_aclnn_exclude,
        ),
        "build_help_summary": _summarize_help_output(help_text),
        "build_help_raw": help_text[:4000],
        "build_commands": {
            "build_help": "bash build.sh -h",
            "compile_ophost": f"bash build.sh -u --ophost --ops='{normalized['op_name']}' --soc='{selected_soc}'",
            "compile_opapi": f"bash build.sh -u --opapi --ops='{normalized['op_name']}' --soc='{selected_soc}'",
            "compile_opkernel": f"bash build.sh -u --opkernel --ops='{normalized['op_name']}' --soc='{selected_soc}'",
            "compile_opkernel_aicpu": f"bash build.sh -u --opkernel --ops='{normalized['op_name']}' --soc='{selected_soc}'",
            "compile_cov_suffix": " --cov",
        },
        "dtype_candidates": dtype_candidates,
        "format_candidates": format_candidates,
        "compile_info_candidates": _compile_info_candidates(op_host_dir),
        "required_attrs": _required_attrs(op_host_dir),
        "reference_ut_paths": _reference_ut_paths(root, normalized["category"], normalized["op_name"]),
        "reference_context": {
            **_shared_test_harness_paths(root),
            "operator_impl_files": _operator_impl_files(root, op_dir),
            "similar_examples": _reference_ut_paths(root, normalized["category"], normalized["op_name"]),
            "current_operator_ut": existing_ut_by_layer,
        },
        "workspace_signatures": _extract_workspace_signatures(op_dir),
        "warnings": normalized["warnings"],
    }
    return result


def inspect_ascend_operator_json(project_root: str | Path, op_path: str, soc_hint: str = "") -> str:
    return json.dumps(
        inspect_ascend_operator(project_root=project_root, op_path=op_path, soc_hint=soc_hint),
        ensure_ascii=False,
        indent=2,
    )
