#!/usr/bin/env python3
"""
Build relative coverage tables for torch and tensorflow.

Relative coverage for module s is:
  (cov(s,e) - min(cov(s))) / (max(cov(s)) - min(cov(s)))

Where cov(s,e) is TestAgent final coverage (from exam_coverage_tables.md).
min/max are taken across all experimental runs, but ONLY using:
  - TestAgent final Coverage (exam_coverage_tables.md)
  - PynguinML Coverage (CSV)

If max == min or cov(s,e) is missing, result is NA.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PERCENT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)%$")


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    v = value.strip()
    if not v or v.upper() == "NA":
        return None
    m = PERCENT_RE.match(v)
    if m:
        return float(m.group(1)) / 100.0
    try:
        return float(v)
    except ValueError:
        return None


def parse_exam_tables(md_path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    data: Dict[str, Dict[str, Optional[float]]] = {
        "torch": {},
        "tensorflow": {},
    }
    if not md_path.exists():
        return data
    section: Optional[str] = None
    headers: Optional[List[str]] = None
    for line in md_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("## "):
            title = line[3:].strip().lower()
            if title.startswith("torch"):
                section = "torch"
            elif title.startswith("tensorflow"):
                section = "tensorflow"
            else:
                section = None
            headers = None
            continue
        if not section or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if headers is None:
            headers = cells
            continue
        if all(c.startswith("---") for c in cells):
            continue
        row = dict(zip(headers, cells))
        target = row.get("TargetModule")
        if not target:
            continue
        coverage = parse_float(row.get("Coverage", ""))
        timeline: List[Optional[float]] = []
        for h in headers:
            if h.startswith("CoverageTimeline_T"):
                timeline.append(parse_float(row.get(h, "")))
        data[section][target] = coverage
    return data


def load_pynguinml(csv_path: Path) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {}
    if not csv_path.exists():
        return values
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row.get("TargetModule", "").strip()
            if not target:
                continue
            val = row.get("Coverage", "")
            cov = parse_float(val or "")
            if cov is None:
                continue
            values.setdefault(target, []).append(cov)
    return values


def compute_relative(
    target: str,
    testagent_cov: Optional[float],
    pynguinml_values: List[float],
) -> Optional[float]:
    all_values: List[float] = []
    if testagent_cov is not None:
        all_values.append(testagent_cov)
    all_values.extend(pynguinml_values)
    if not all_values or testagent_cov is None:
        return None
    min_v = min(all_values)
    max_v = max(all_values)
    if max_v == min_v:
        return None
    return (testagent_cov - min_v) / (max_v - min_v)


def build_table(rows: List[Tuple[str, Optional[float]]]) -> str:
    lines = [
        "| TargetModule | RelativeCoverage |",
        "| --- | --- |",
    ]
    for target, value in rows:
        if value is None:
            val = "NA"
        else:
            val = f"{value:.4f}"
        lines.append(f"| {target} | {val} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam-table", default="exam_coverage_tables.md")
    parser.add_argument("--torch-csv", default="pynguinml_artifact/data/torch-11042025.csv")
    parser.add_argument("--tensorflow-csv", default="pynguinml_artifact/data/tensorflow-11042025.csv")
    parser.add_argument("--out", default="", help="Output markdown file")
    args = parser.parse_args()

    exam_tables = parse_exam_tables(Path(args.exam_table))
    torch_py = load_pynguinml(Path(args.torch_csv))
    tf_py = load_pynguinml(Path(args.tensorflow_csv))

    torch_rows: List[Tuple[str, Optional[float]]] = []
    for target, cov in sorted(exam_tables["torch"].items()):
        rel = compute_relative(target, cov, torch_py.get(target, []))
        torch_rows.append((target, rel))

    tf_rows: List[Tuple[str, Optional[float]]] = []
    for target, cov in sorted(exam_tables["tensorflow"].items()):
        rel = compute_relative(target, cov, tf_py.get(target, []))
        tf_rows.append((target, rel))

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
