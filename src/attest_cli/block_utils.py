"""
Helpers for working with semantic BLOCK markers across multiple file types.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


BLOCK_MARKER_RE = re.compile(
    r"^\s*(?P<prefix>#|//)\s*====\s*BLOCK:(?P<block_id>[A-Za-z0-9_]+)"
    r"(?P<suffix>\s+(START|END))?\s*====\s*$"
)


def detect_comment_style(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"}:
        return "//"
    return "#"


def placeholder_marker(block_id: str, comment_style: str) -> str:
    return f"{comment_style} ==== BLOCK:{block_id} ===="


def start_marker(block_id: str, comment_style: str) -> str:
    return f"{comment_style} ==== BLOCK:{block_id} START ===="


def end_marker(block_id: str, comment_style: str) -> str:
    return f"{comment_style} ==== BLOCK:{block_id} END ===="


def build_block_entries(path: str | Path) -> List[Dict[str, Any]]:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return []

    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    blocks: Dict[str, Dict[str, Any]] = {}
    for line_num, line in enumerate(lines, 1):
        match = BLOCK_MARKER_RE.match(line)
        if not match:
            continue

        block_id = match.group("block_id")
        prefix = match.group("prefix")
        suffix = match.group("suffix") or ""
        entry = blocks.setdefault(
            block_id,
            {
                "block_id": block_id,
                "comment_style": prefix,
                "start_line": None,
                "end_line": None,
                "markers": [],
            },
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

    entries: List[Dict[str, Any]] = []
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
                "comment_style": entry["comment_style"],
                "start_line": start,
                "end_line": end,
                "status": status,
            }
        )

    entries.sort(key=lambda item: (item["start_line"] or 0, item["block_id"]))
    return entries


def build_block_index_json(path: str | Path) -> str:
    entries = build_block_entries(path)
    if not entries:
        return "N/A (no block markers found)"
    return json.dumps(entries, ensure_ascii=True, indent=2)
