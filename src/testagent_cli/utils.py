import shutil
from pathlib import Path
from typing import Iterable


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def backup_file(path: Path) -> Path:
    """Create a .bak copy of a file before modification. Skip if directory or doesn't exist."""
    if not path.exists() or path.is_dir():
        return path
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    return backup


def format_bullet(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def slugify_target(name: str) -> str:
    """
    Convert a fully-qualified name (e.g., package.module:func) into a filesystem-safe slug.
    """
    import re

    slug = re.sub(r"[:\\.]+", "_", name)
    slug = re.sub(r"[^0-9a-zA-Z_]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "target"
