import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CONFIG_DIR


# 默认全局 sessions 目录，兼容旧行为
DEFAULT_SESSIONS_DIR = CONFIG_DIR / "sessions"
DEFAULT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _sessions_dir(workspace: Optional[str | Path] = None) -> Path:
    """
    返回日志目录：优先写入 workspace/.attest/logs，否则落到全局默认目录。
    """
    if workspace:
        base = Path(workspace) / ".attest" / "logs"
    else:
        base = DEFAULT_SESSIONS_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def session_path(session_id: str, workspace: Optional[str | Path] = None) -> Path:
    return _sessions_dir(workspace) / f"{session_id}.jsonl"


def list_sessions(workspace: Optional[str | Path] = None) -> List[str]:
    return sorted(p.stem for p in _sessions_dir(workspace).glob("*.jsonl"))


def clear_session(session_id: str, workspace: Optional[str | Path] = None) -> None:
    path = session_path(session_id, workspace)
    if path.exists():
        path.unlink()


def append_message(
    session_id: str,
    role: str,
    content: Any,
    workspace: Optional[str | Path] = None,
    stage: Optional[str] = None,
) -> None:
    """
    追加一条消息到日志。

    Args:
        session_id: 会话/工作流 ID
        role: user/assistant/tool 等
        content: 任意内容（字符串或结构化数据）
        workspace: 如提供则写入 workspace/.attest/logs
        stage: 可选，标记所在阶段（工作流使用）
    """
    rec: Dict[str, Any] = {"role": role, "content": content}
    if stage:
        rec["stage"] = stage
    with session_path(session_id, workspace).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_history(session_id: str, workspace: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    path = session_path(session_id, workspace)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
