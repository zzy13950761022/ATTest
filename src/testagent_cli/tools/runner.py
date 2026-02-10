from typing import Dict, List, Optional, Type

from .base import Tool, ToolContext, ToolResult
from .builtin import (
    ListFilesTool,
    ReadFileTool,
    PartReadTool,
    SearchTool,
    WriteFileTool,
    ReplaceInFileTool,
    ReplaceBlockTool,
    ExecCommandTool,
    InspectPythonTool,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Tool] = {}

    def register(self, tool_cls: Type[Tool]) -> None:
        tool = tool_cls()
        self._registry[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._registry.get(name)

    def all(self) -> Dict[str, Tool]:
        return self._registry
    
    def to_llm_schema(self) -> List[Dict]:
        """
        Convert all registered tools to OpenAI function calling schema.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        schemas = []
        for tool in self._registry.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return schemas



class ToolRunner:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, name: str, params: Dict, ctx: ToolContext) -> ToolResult:
        tool = self.registry.get(name)
        if not tool:
            return ToolResult(False, "", f"Unknown tool: {name}")
        
        # Import UI component
        try:
            from ..ui.tool_display import tool_display
            use_rich = True
        except ImportError:
            use_rich = False
        
        # Show tool call with rich UI
        if use_rich:
            tool_display.show_call(name, params, readonly=tool.readonly)
        else:
            print(f"🔧 Calling: {name} with params={params}")
        
        # Check approval for write operations
        if not tool.readonly and not ctx.auto_approve:
            approved = self._ask_approve(tool, params)
            if not approved:
                result = ToolResult(False, "", "Denied by user")
                if use_rich:
                    tool_display.show_result(False, "", "Denied by user")
                return result
        
        # Execute tool
        result = tool.execute(params, ctx)
        
        # Show result with rich UI
        if use_rich:
            tool_display.show_result(result.ok, result.output, result.error)
        else:
            if result.ok:
                print(f"✅ Success: {result.output[:100]}")
            else:
                print(f"❌ Error: {result.error}")
        
        return result

    def _ask_approve(self, tool: Tool, params: Dict) -> bool:
        # Import console for rich display
        try:
            from ..ui.console import console
            from rich.panel import Panel
            console.print()
            console.print(Panel(
                f"Tool: [bold]{tool.name}[/bold]\nParameters: {params}",
                title="[yellow]⚠ Approval Required[/yellow]",
                border_style="yellow"
            ))
            resp = console.input("[yellow]Approve this operation? (y/n):[/yellow] ").strip().lower()
        except ImportError:
            print(f"[APPROVAL] Tool '{tool.name}' params={params}")
            resp = input("Approve? (y/n): ").strip().lower()
        return resp in {"y", "yes"}


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for cls in [
        ListFilesTool,
        ReadFileTool,
        PartReadTool,
        SearchTool,
        WriteFileTool,
        ReplaceInFileTool,
        ReplaceBlockTool,
        ExecCommandTool,
        InspectPythonTool,
    ]:
        reg.register(cls)
    return reg
