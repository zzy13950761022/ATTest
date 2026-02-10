#!/usr/bin/env python3
"""
Helper script for inspecting Python objects in a subprocess.
This allows inspection in a different Python environment than the one running testagent.
"""
import sys
import json
import importlib
import inspect
from pathlib import Path


def inspect_object(target: str, add_cwd_to_path: bool = True, max_source_length: int = 4000):
    """Inspect a Python object and return metadata as JSON."""

    def _split_target(name: str):
        if ":" in name:
            module_name, attr_path = name.split(":", 1)
        else:
            parts = name.split(".")
            if len(parts) < 2:
                raise ValueError("Target must include module path")
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
        # Add cwd to path if requested
        if add_cwd_to_path:
            cwd = str(Path.cwd().resolve())
            if cwd not in sys.path:
                sys.path.insert(0, cwd)

        # Load the object
        module, obj = _load_object(target)
        obj_is_module = inspect.ismodule(obj)
        module_for_metadata = obj if obj_is_module else module

        # Gather metadata
        info = {
            "target": target,
            "module": module_for_metadata.__name__,
            "module_file": getattr(module_for_metadata, "__file__", None),
            "qualname": getattr(obj, "__qualname__", None),
            "type": type(obj).__name__,
            "docstring": inspect.getdoc(obj) or "",
            "module_docstring": inspect.getdoc(module) or "",
            "python_version": sys.version,
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

        # Source code
        try:
            source = inspect.getsource(obj)
            if max_source_length and len(source) > max_source_length:
                source = source[:max_source_length] + "\n# ... truncated ..."
            info["source"] = source
        except Exception:
            info["source"] = None

        # If it's a module, get public members
        if obj_is_module:
            try:
                members = []
                for name in dir(obj):
                    if not name.startswith("_"):
                        try:
                            member = getattr(obj, name)
                            members.append({
                                "name": name,
                                "type": type(member).__name__
                            })
                        except Exception:
                            pass
                info["public_members"] = members[:50]  # Limit to 50
            except Exception:
                pass

        return {"success": True, "data": info}

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Missing target argument"}))
        sys.exit(1)

    target = sys.argv[1]
    add_cwd = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else True
    max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

    result = inspect_object(target, add_cwd, max_len)
    print(json.dumps(result, indent=2))
