#!/usr/bin/env python3
"""
Post-hoc fix for coverage_valid logic.
Recomputes coverage_valid and overall averages using the new baseline=0% skip logic.
Run after v8 completes to update all coverage summaries.

Usage:
    python3 tools/fix_coverage_valid.py [--dry-run] [workspace_root...]

Defaults to /mnt/fangcr/workspace-qwen-3.7-plus_v8/
"""
import argparse
import json
from pathlib import Path


def recompute_coverage(cs: dict) -> dict:
    runs = cs.get("runs", {})
    baseline = runs.get("baseline", {})
    generated = runs.get("generated", {})
    bl_pl = baseline.get("per_layer") or {}
    gen_pl = generated.get("per_layer") or {}

    per_layer = cs.get("per_layer", {})
    layers = list(per_layer.keys()) or list(set(list(bl_pl.keys()) + list(gen_pl.keys())))

    baseline_values = []
    generated_values = []
    invalid_layers = {}
    skipped_layers = {}

    for layer in layers:
        bl_meta = bl_pl.get(layer, {})
        gen_meta = gen_pl.get(layer, {})
        bl_val = float(bl_meta.get("line_coverage", 0.0))
        gen_val = float(gen_meta.get("line_coverage", 0.0))
        bl_valid = bool(bl_meta.get("coverage_valid", True))
        gen_valid = bool(gen_meta.get("coverage_valid", True))
        baseline_uncovered = not bl_valid and bl_val == 0.0

        if bl_valid and gen_valid:
            baseline_values.append(bl_val)
            generated_values.append(gen_val)
        elif baseline_uncovered:
            skipped_layers[layer] = "baseline has no instrumented code for this layer"
            if gen_valid and bl_val > 0:
                baseline_values.append(bl_val)
                generated_values.append(gen_val)
        else:
            invalid_layers[layer] = {
                "baseline": str(bl_meta.get("coverage_error_reason") or ""),
                "generated": str(gen_meta.get("coverage_error_reason") or ""),
            }

    bl_avg = sum(baseline_values) / len(baseline_values) if baseline_values else 0.0
    gen_avg = sum(generated_values) / len(generated_values) if generated_values else 0.0
    coverage_valid = not invalid_layers and bool(baseline_values)

    cs["coverage_valid"] = coverage_valid
    cs["coverage_errors"] = invalid_layers
    cs["skipped_layers"] = skipped_layers
    cs.setdefault("overall", {})
    cs["overall"]["baseline_avg"] = round(bl_avg, 4)
    cs["overall"]["generated_avg"] = round(gen_avg, 4)
    cs["overall"]["delta_line_coverage"] = round(gen_avg - bl_avg, 4)
    cs["overall"]["coverage_valid"] = coverage_valid
    return cs


def process_workspace(root: Path, dry_run: bool = False):
    updated = []
    unchanged = []
    not_found = 0

    for d in sorted(root.glob("attest-*-compare")):
        cs_file = d / ".attest" / "artifacts" / "execute_tests" / "current_coverage_summary.json"
        if not cs_file.exists():
            not_found += 1
            continue

        original_text = cs_file.read_text()
        data = json.loads(original_text)
        old_valid = data.get("coverage_valid")
        old_delta = data.get("overall", {}).get("delta_line_coverage", 0)

        recompute_coverage(data)

        new_valid = data.get("coverage_valid")
        new_delta = data.get("overall", {}).get("delta_line_coverage", 0)

        op = d.name.replace("attest-", "").replace("-compare", "")

        changed = old_valid != new_valid or abs(old_delta - new_delta) > 0.01
        if changed or "skipped_layers" not in data:
            if not dry_run:
                backup = cs_file.with_suffix(".json.bak")
                if not backup.exists():
                    backup.write_text(original_text)
                cs_file.write_text(json.dumps(data, indent=2))
            updated.append(f"  {op}: valid={old_valid}->{new_valid} delta={old_delta:+.2f}->{new_delta:+.2f}pp")
        else:
            unchanged.append(f"  {op}: valid={new_valid} delta={new_delta:+.2f}pp")

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}{root}: updated {len(updated)}, unchanged {len(unchanged)}, no-data {not_found}")
    if updated:
        print("Updated:")
        for u in updated:
            print(u)
    if unchanged:
        print("Unchanged:")
        for u in unchanged:
            print(u)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying files")
    parser.add_argument("workspaces", nargs="*", default=["/mnt/fangcr/workspace-qwen-3.7-plus_v8"])
    args = parser.parse_args()

    for ws in args.workspaces:
        process_workspace(Path(ws), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
