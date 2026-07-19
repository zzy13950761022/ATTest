import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CONFIG_DIR


# Default global sessions directory for backward compatibility
DEFAULT_SESSIONS_DIR = CONFIG_DIR / "sessions"
DEFAULT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _sessions_dir(workspace: Optional[str | Path] = None) -> Path:
    """
    Return the log directory: prefer `workspace/.attest/logs`; otherwise fall back to the global default directory.
    """
    if workspace:
        workspace_path = Path(workspace)
        preferred = workspace_path / ".attest" / "logs"
        legacy = workspace_path / ".testagent" / "logs"
        base = preferred if preferred.exists() or not legacy.exists() else legacy
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
    Append a message to the log.

    Args:
        session_id: Session / workflow ID
        role: user / assistant / tool, and so on
        content: Arbitrary content (string or structured data)
        workspace: If provided, write to `workspace/.attest/logs`
        stage: Optional stage marker (used by the workflow)
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
