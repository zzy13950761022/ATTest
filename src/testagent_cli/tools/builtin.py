import os
import subprocess
from pathlib import Path
from typing import Any, Dict
import importlib
import inspect
import json
import sys

from .base import Tool, ToolContext, ToolResult
from ..utils import backup_file, ensure_parent


class ListFilesTool(Tool):
    name = "list_files"
    readonly = True
    description = "List files under cwd (non-recursive)."

    def __init__(self):
        super().__init__()
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (defaults to cwd)"
                }
            }
        }

    def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_path = params.get("path")
        if raw_path:
            candidate = Path(raw_path)
            path = candidate if candidate.is_absolute() else Path(ctx.cwd) / candidate
        else:
            path = Path(ctx.cwd)
        if not path.exists():
            return ToolResult(False, "", f"Path not found: {path}")
        entries = []
        for p in sorted(path.iterdir()):
            suffix = "/" if p.is_dir() else ""
            entries.append(str(p.name) + suffix)
        return ToolResult(True, "\n".join(entries))


class ReadFileTool(Tool):
    name = "read_file"
    readonly = True
    description = "Read file content."

    def __init__(self):
        super().__init__()
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to file to read"
                }
            },
            "required": ["path"]
        }

    def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = Path(ctx.cwd) / params.get("path", "")
        if not path.exists():
            return ToolResult(False, "", f"File not found: {path}")
        try:
            content = path.read_text(encoding="utf-8")
            return ToolResult(True, content)
        except UnicodeDecodeError:
            return ToolResult(False, "", f"File not utf-8: {path}")


class SearchTool(Tool):
    name = "search"
    readonly = True
    description = "Search for text pattern in files (recursive)."

    def __init__(self):
        super().__init__()
        self.parameters = {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (plain text or regex)"
                },
                "path": {
                    "type": "string",
                    "description": "Path to search in (defaults to current directory)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter (e.g., '*.py', '*.cpp')"
                },
                "regex": {
                    "type": "boolean",
                    "description": "Treat pattern as regex (default: False)"
                }
            },
            "required": ["pattern"]
        }

    def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = params.get("pattern")
        target_path = params.get("path", ".")
        file_pattern = params.get("file_pattern", "*")
        use_regex = params.get("regex", False)
        
        if not pattern:
            return ToolResult(False, "", "Missing pattern")
        
        search_dir = Path(ctx.cwd) / target_path
        if not search_dir.exists():
            return ToolResult(False, "", f"Path not found: {search_dir}")
        
        # Prepare regex pattern
        import re
        if use_regex:
            try:
                regex_pattern = re.compile(pattern)
            except re.error as e:
                return ToolResult(False, "", f"Invalid regex: {e}")
        else:
            regex_pattern = re.compile(re.escape(pattern))
        
        # Search helper for a single file
        def _search_file(file_path: Path, results: list) -> None:
            if not file_path.is_file():
                return
            if any(part.startswith('.') for part in file_path.parts):
                return
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            except (UnicodeDecodeError, PermissionError):
                return
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex_pattern.search(line):
                    relative_path = file_path.relative_to(ctx.cwd)
                    results.append(f"{relative_path}:{line_num}:{line.strip()}")
                    if len(results) >= 100:
                        break
        
        results = []
        try:
            if search_dir.is_file():
                _search_file(search_dir, results)
            else:
                for file_path in search_dir.rglob(file_pattern):
                    _search_file(file_path, results)
                    if len(results) >= 100:
                        break
            
            if not results:
                return ToolResult(True, f"No matches found for '{pattern}'")
            
            return ToolResult(True, "\n".join(results[:100]))
        except Exception as e:
            return ToolResult(False, "", f"Search failed: {e}")


class WriteFileTool(Tool):
    name = "write_file"
    readonly = False
    description = "Write content to file with backup."

    def __init__(self):
        super().__init__()
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to file"
                }
            },
            "required": ["path", "content"]
        }

    def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        file_path = params.get("path", "")
        content = params.get("content", "")
        
        # Validate path
        if not file_path:
            return ToolResult(False, "", "Error: 'path' parameter is required and cannot be empty")
        
        # Reject absolute paths
        if Path(file_path).is_absolute():
            return ToolResult(False, "", f"Error: Absolute paths are not allowed. Use relative path instead of: {file_path}")
        
        path = Path(ctx.cwd) / file_path
        
        # Don't allow writing to directories
        if path.exists() and path.is_dir():
            return ToolResult(False, "", f"Error: Path is a directory, not a file: {file_path}")
        
        ensure_parent(path)
        backup_file(path)
        path.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Wrote {path}")


class ReplaceInFileTool(Tool):
    name = "replace_in_file"
    readonly = False
    description = "Replace text in file (first occurrence)."

    def __init__(self):
        super().__init__()
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to file"
                },
                "search": {
                    "type": "string",
                    "description": "Text to search for"
                },
                "replace": {
                    "type": "string",
                    "description": "Text to replace with"
                }
            },
            "required": ["path", "search", "replace"]
        }

    def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = Path(ctx.cwd) / params.get("path", "")
        search = params.get("search")
        replace = params.get("replace", "")
        if not path.exists():
            return ToolResult(False, "", f"File not found: {path}")
        if search is None:
            return ToolResult(False, "", "Missing search text")
        text = path.read_text(encoding="utf-8")
        if search not in text:
            return ToolResult(False, "", "Search text not found")
        backup_file(path)
        path.write_text(text.replace(search, replace, 1), encoding="utf-8")
        return ToolResult(True, f"Replaced first occurrence in {path}")


class ExecCommandTool(Tool):
    name = "exec_command"
    readonly = False
    description = "Execute a shell command (dangerous; requires approval)."

    def __init__(self):
        super().__init__()
        self.parameters = {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["cmd"]
        }

    def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        cmd = params.get("cmd")
        if not cmd:
            return ToolResult(False, "", "Missing cmd")
        try:
            completed = subprocess.run(
                cmd,
                cwd=ctx.cwd,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )
            output = completed.stdout + completed.stderr
            ok = completed.returncode == 0
            return ToolResult(ok, output, None if ok else f"exit {completed.returncode}")
        except Exception as e:  # pragma: no cover - broad safety
            return ToolResult(False, "", str(e))


class InspectPythonTool(Tool):
    name = "inspect_python"
    readonly = True
    description = "Inspect a Python object via fully qualified name and return metadata."

    def __init__(self):
        super().__init__()
        self.parameters = {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Fully qualified name, e.g., 'pkg.module:func' or 'pkg.module.Class.method'"
                },
                "add_cwd_to_path": {
                    "type": "boolean",
                    "description": "Whether to prepend cwd to sys.path before import (default: true)"
                },
                "max_source_length": {
                    "type": "integer",
                    "description": "Optional limit for returned source length (characters)"
                }
            },
            "required": ["target"]
        }

    def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = params.get("target")
        if not target:
            return ToolResult(False, "", "Missing target")

        add_cwd = params.get("add_cwd_to_path", True)
        max_source_len = params.get("max_source_length") or 4000
        added_path = False
        cwd_path = str(Path(ctx.cwd).resolve())

        def _split_target(name: str) -> tuple[str, str]:
            if ":" in name:
                module_name, attr_path = name.split(":", 1)
            else:
                parts = name.split(".")
                if len(parts) < 2:
                    raise ValueError("Target must include module path, e.g., 'pkg.module:func'")
                module_name, attr_path = ".".join(parts[:-1]), parts[-1]
            return module_name, attr_path

        def _load_object(name: str):
            module_name, attr_path = _split_target(name)
            module = importlib.import_module(module_name)
            obj = module
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            return module, obj

        try:
            if add_cwd and cwd_path not in sys.path:
                sys.path.insert(0, cwd_path)
                added_path = True

            module, obj = _load_object(target)
            obj_is_module = inspect.ismodule(obj)
            # Prefer the object's own module info when the target itself is a module
            module_for_metadata = obj if obj_is_module else module
            info: Dict[str, Any] = {
                "target": target,
                "module": module_for_metadata.__name__,
                "module_file": getattr(module_for_metadata, "__file__", None),
                "qualname": getattr(obj, "__qualname__", None),
                "type": type(obj).__name__,
                "docstring": inspect.getdoc(obj) or "",
                "module_docstring": inspect.getdoc(module) or "",
            }

            # Signature for callables
            sig = None
            callable_obj = obj
            if inspect.isclass(obj) and hasattr(obj, "__call__"):
                callable_obj = obj.__call__
            if callable(callable_obj):
                try:
                    sig = str(inspect.signature(callable_obj))
                except (TypeError, ValueError):
                    sig = None
            if sig:
                info["signature"] = sig

            # Type hints
            try:
                hints = inspect.get_annotations(obj, eval_str=False)
                if hints:
                    info["annotations"] = {k: str(v) for k, v in hints.items()}
            except Exception:
                pass

            # Source code (trimmed)
            try:
                source = inspect.getsource(obj)
                if max_source_len and len(source) > max_source_len:
                    source = source[:max_source_len] + "\n# ... truncated ..."
                info["source"] = source
            except (OSError, TypeError):
                pass

            return ToolResult(True, json.dumps(info, ensure_ascii=False, indent=2))
        except Exception as e:
            return ToolResult(False, "", f"{type(e).__name__}: {e}")
        finally:
            if added_path and cwd_path in sys.path:
                try:
                    sys.path.remove(cwd_path)
                except ValueError:
                    pass
