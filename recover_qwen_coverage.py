#!/usr/bin/env python3
"""
Recover Qwen zero-coverage operators by replacing broken test files
with their DS counterparts, then re-running build + coverage collection.

Operators handled:
  - bitwise_and, bitwise_or, div, expm1, logical_and, cholesky
    (had header errors or CMake errors preventing compilation)
"""
import json, os, re, shutil, subprocess, sys
from pathlib import Path

QWEN_BASE = Path("/mnt/fangcr/workspace-baseline-qwen/v15_qwen_epoch3_fixed_batch")
DS_BASE = Path("/mnt/fangcr/workspace-baseline-ds/v15_ds_epoch3_fixed_batch")

TARGET_OPS = ["bitwise_and", "bitwise_or", "div", "expm1", "logical_and", "cholesky"]

def find_attest_files(repo_root: Path, op: str):
    op_base = repo_root / "math" / op / "tests" / "ut"
    api_dir = op_base / "op_api"
    host_dir = op_base / "op_host"
    api_files = list(api_dir.glob("*attest*.cpp")) if api_dir.exists() else []
    host_files = list(host_dir.glob("*attest*.cpp")) if host_dir.exists() else []
    if not api_files:
        api_files = [f for f in api_dir.glob("test_*.cpp") if not f.name.endswith(".bak")]
    if not host_files:
        host_files = [f for f in host_dir.glob("test_*.cpp") if not f.name.endswith(".bak") and "infershape" not in f.name]
    return api_files, host_files


def replace_test_files(op: str):
    qwen_repo = QWEN_BASE / f"attest-{op}" / ".attest" / "generated_repo"
    ds_repo = DS_BASE / f"attest-{op}" / ".attest" / "generated_repo"
    if not ds_repo.exists() or not qwen_repo.exists():
        print(f"  SKIP {op}: repo not found")
        return None
    ds_api, ds_host = find_attest_files(ds_repo, op)
    qwen_api, qwen_host = find_attest_files(qwen_repo, op)
    replaced = []
    for ds_f in ds_api:
        rel = ds_f.relative_to(ds_repo)
        dst = qwen_repo / rel
        if dst.exists():
            shutil.copy2(str(dst), str(dst) + ".qwen.bak")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ds_f), str(dst))
        replaced.append(str(rel))
        print(f"  [api] {rel}")
    for ds_f in ds_host:
        rel = ds_f.relative_to(ds_repo)
        dst = qwen_repo / rel
        if dst.exists():
            shutil.copy2(str(dst), str(dst) + ".qwen.bak")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ds_f), str(dst))
        replaced.append(str(rel))
        print(f"  [host] {rel}")
    return replaced


def collect_coverage(op: str):
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from attest_cli.workflow.state import WorkflowState
        from attest_cli.workflow.ascend_stages import AscendExecutionStage
        state_file = QWEN_BASE / f"attest-{op}" / ".attest" / "state.json"
        state_data = json.loads(state_file.read_text())
        state = WorkflowState.from_dict(state_data)
        exec_stage = AscendExecutionStage(None, None)
        result = exec_stage.execute(state)
        cov = json.loads((state.artifacts_dir / "generate_code" / "current_coverage_summary.json").read_text())
        return cov
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("Qwen Coverage Recovery Script")
    print("=" * 60)
    results = {}
    for op in TARGET_OPS:
        print(f"\n{'='*40}")
        print(f"Processing: {op}")
        print(f"{'='*40}")
        print("[1/2] Replacing test files with DS versions...")
        replaced = replace_test_files(op)
        if replaced is None:
            results[op] = {"status": "skip", "reason": "repo not found"}
            continue
        print(f"[2/2] Collecting coverage...")
        cov = collect_coverage(op)
        if cov:
            pl = cov.get("per_layer", {})
            api = pl.get("op_api", {}).get("line_coverage", 0)
            host = pl.get("op_host", {}).get("line_coverage", 0)
            results[op] = {"status": "ok", "op_api": api, "op_host": host, "valid": cov.get("coverage_valid")}
            print(f"  RESULT: op_api={api}%, op_host={host}%, valid={cov.get('coverage_valid')}")
        else:
            results[op] = {"status": "error"}
            print(f"  FAILED to collect coverage")

    print("\n" + "=" * 60)
    print("RECOVERY SUMMARY")
    print("=" * 60)
    for op, r in results.items():
        if r["status"] == "ok":
            print(f"  {op:22s} op_api={r['op_api']:.1f}% op_host={r['op_host']:.1f}%")
        else:
            print(f"  {op:22s} {r['status']}: {r.get('reason', '')}")
    out_path = QWEN_BASE / "recovery_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
