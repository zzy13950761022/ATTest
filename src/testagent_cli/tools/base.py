from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolContext:
    cwd: str
    auto_approve: bool = False


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: Optional[str] = None


class Tool:
    name: str = ""
    readonly: bool = True
    description: str = ""
    parameters: Dict[str, Any] = None  # JSONSchema for tool input

    def __init__(self):
        """Override in subclass to set parameters if needed."""
        if self.parameters is None:
            self.parameters = {"type": "object", "properties": {}}

    def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError

