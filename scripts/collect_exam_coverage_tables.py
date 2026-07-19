#!/usr/bin/env python3
"""
Generate markdown tables for exam/torch and exam/tensorflow.

Outputs two tables with columns:
  - TargetModule
  - TotalTime (seconds)
  - Coverage (final branch coverage for tests/ from coverage.xml)
  - CoverageTimeline_T* (per-epoch branch coverage from v*_execution_log.txt)

Timeline values use per-epoch coverage.xml snapshots if found. If not found,
the script falls back to a heuristic derived from the coverage table in the
execution log (Branch/BrPart). This may differ from exact branch-rate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")
COND_RE = re.compile(r"\((\d+)/(\d+)\)")
LOG_COVERAGE_HEADER = "Name"
LOG_COVERAGE_TAIL = "Coverage XML written to file"


def parse_iso(ts: str) -> Optional[dt.datetime]:
    if not ts or not ISO_RE.match(ts):
        return None
    try:
        return dt.datetime.fromisoformat(ts)
    except ValueError:
        return None


def total_time_seconds(state_path: Path) -> Optional[float]:
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text())
    except Exception:
        return None
    times: List[dt.datetime] = []
    created_at = parse_iso(data.get("created_at", ""))
    if created_at:
        times.append(created_at)
    for item in data.get("stage_history", []) or []:
        ts = parse_iso(item.get("timestamp", ""))
        if ts:
            times.append(ts)
    if not times:
        return None
    return (max(times) - min(times)).total_seconds()


def branch_rate_from_coverage_xml(xml_path: Path) -> Optional[float]:
    if not xml_path.exists():
        return None
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    branches_valid = 0
    branches_covered = 0
    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename", "")
        if not filename.startswith("tests/"):
            continue
        for line in cls.findall(".//line"):
            if line.attrib.get("branch") != "true":
                continue
            cc = line.attrib.get("condition-coverage", "")
            m = COND_RE.search(cc)
            if not m:
                continue
            branches_covered += int(m.group(1))
            branches_valid += int(m.group(2))
    if branches_valid == 0:
        return None
    return branches_covered / branches_valid


def extract_coverage_table(log_text: str) -> List[str]:
    if LOG_COVERAGE_HEADER not in log_text:
        return []
    start = log_text.rfind(LOG_COVERAGE_HEADER)
    if start == -1:
        return []
    end = log_text.find(LOG_COVERAGE_TAIL, start)
    if end == -1:
        end = len(log_text)
    section = log_text[start:end]
    lines = [ln for ln in section.splitlines() if ln.strip()]
    if not lines:
        return []
    # Remove divider lines
    return [ln for ln in lines if not ln.startswith("---")]


def branch_rate_from_log(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text()
    except Exception:
        return None
    lines = extract_coverage_table(text)
    if not lines or len(lines) < 2:
        return None
    rows = lines[1:]
    branches = 0
    brpart = 0
    for row in rows:
        parts = row.split()
        if not parts:
            continue
        name = parts[0].replace("\\", "/")
        if name == "TOTAL":
            continue
        if not name.startswith("tests/"):
            continue
        if len(parts) < 6:
            continue
        try:
            branches += int(parts[3])
            brpart += int(parts[4])
        except ValueError:
            continue
    if branches == 0:
        return None
    # Heuristic: treat BrPart as uncovered branches.
    return max(0.0, (branches - brpart) / branches)


def find_epoch_logs(execute_dir: Path) -> List[Tuple[int, Path]]:
    logs = []
    for p in execute_dir.glob("v*_execution_log.txt"):
        m = re.match(r"v(\d+)_execution_log\.txt", p.name)
        if not m:
            continue
        logs.append((int(m.group(1)), p))
    return sorted(logs, key=lambda x: x[0])


def find_epoch_coverage_xml(module_dir: Path, epoch: int) -> Optional[Path]:
    # If a per-epoch coverage xml snapshot exists, prefer it.
    candidates = [
        module_dir / f"coverage_v{epoch}.xml",
        module_dir / f"coverage_{epoch}.xml",
        module_dir / ".testagent" / "artifacts" / "execute_tests" / f"v{epoch}_coverage.xml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_module_row(module_dir: Path, prefix: str) -> Dict[str, object]:
    target_module = f"{prefix}.{module_dir.name}"
    state_path = module_dir / ".testagent" / "state.json"
    total_time = total_time_seconds(state_path)

    coverage_xml = module_dir / "coverage.xml"
    final_branch = branch_rate_from_coverage_xml(coverage_xml)

    timeline: Dict[int, Optional[float]] = {}
    execute_dir = module_dir / ".testagent" / "artifacts" / "execute_tests"
    if execute_dir.exists():
        for epoch, log_path in find_epoch_logs(execute_dir):
            cov_xml = find_epoch_coverage_xml(module_dir, epoch)
            if cov_xml:
                timeline[epoch] = branch_rate_from_coverage_xml(cov_xml)
            else:
                timeline[epoch] = branch_rate_from_log(log_path)

    return {
        "TargetModule": target_module,
        "TotalTime": total_time,
        "Coverage": final_branch,
        "Timeline": timeline,
    }


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.2f}%"


def format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.1f}"


def build_table(rows: List[Dict[str, object]]) -> str:
    # Determine max epoch count across rows.
    max_epoch = 0
    for r in rows:
        if r["Timeline"]:
            max_epoch = max(max_epoch, max(r["Timeline"].keys()))
    headers = ["TargetModule", "TotalTime(s)", "Coverage"] + [
        f"CoverageTimeline_T{idx}" for idx in range(1, max_epoch + 1)
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        timeline = r["Timeline"]
        row = [
            str(r["TargetModule"]),
            format_seconds(r["TotalTime"]),
            format_pct(r["Coverage"]),
        ]
        for idx in range(1, max_epoch + 1):
            row.append(format_pct(timeline.get(idx)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def collect(base_dir: Path, prefix: str) -> List[Dict[str, object]]:
    rows = []
    for item in sorted(base_dir.iterdir()):
        if not item.is_dir():
            continue
        # Skip non-module folders
        if item.name.startswith("."):
            continue
        rows.append(collect_module_row(item, prefix))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="exam", help="Root exam directory")
    parser.add_argument("--out", default="", help="Output markdown file")
    args = parser.parse_args()

    root = Path(args.root)
    torch_rows = collect(root / "torch", "torch")
    tf_rows = collect(root / "tensorflow", "tensorflow")

    output = []
    output.append("## Torch")
    output.append(build_table(torch_rows))
    output.append("")
    output.append("## TensorFlow")
    output.append(build_table(tf_rows))
    output_text = "\n".join(output).strip() + "\n"

    if args.out:
        Path(args.out).write_text(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
