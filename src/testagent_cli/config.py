import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


CONFIG_DIR = Path(os.environ.get("TESTAGENT_CONFIG_DIR", Path.home() / ".testagent_cli"))
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "config.json"


DEFAULT_CONFIG: Dict[str, Any] = {
    "api": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-18265a21755744019548b3cadd12ddff", # temp
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "preferences": {
        "auto_approve": False,
    },
    # Python 项目配置
    "project": {
        "root": ".",
        "test_file_template": "tests/test_{target_slug}.py",
        "build_dir": "",
        "output_binary_template": "",
        "python_executable": ""  # 可选：指定用于检查/执行测试的 Python 解释器路径（如 /path/to/venv/bin/python）
    },
    # 测试执行命令配置（默认使用 pytest）
    "commands": {
        "compile": "",
        "install": "",
        "run_test": "PYTHONPATH={project_root}:$PYTHONPATH pytest -q {test_file_path}"
    },
}



def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = DEFAULT_CONFIG.copy()
    return _merge_defaults(data, DEFAULT_CONFIG)


def save_config(cfg: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def set_config_value(key: str, value: Any) -> None:
    cfg = load_config()
    _set_nested(cfg, key.split("."), value)
    save_config(cfg)


def get_config_value(key: str) -> Optional[Any]:
    cfg = load_config()
    return _get_nested(cfg, key.split("."))


def list_config() -> Dict[str, Any]:
    return load_config()


def _merge_defaults(current: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = defaults.copy()
    for k, v in current.items():
        if isinstance(v, dict) and isinstance(defaults.get(k), dict):
            merged[k] = _merge_defaults(v, defaults[k])  # type: ignore[arg-type]
        else:
            merged[k] = v
    return merged


def _set_nested(d: Dict[str, Any], keys: list[str], value: Any) -> None:
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _get_nested(d: Dict[str, Any], keys: list[str]) -> Optional[Any]:
    node: Any = d
    for k in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(k)
    return node
