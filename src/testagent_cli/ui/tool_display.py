"""
Tool call display components with rich formatting.
"""
import time
from typing import Dict, Any, Optional
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from .console import console
from .theme import STYLE_DIM


class ToolCallDisplay:
    """
    Display tool calls with professional formatting.
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
    
    def show_call(self, tool_name: str, params: Dict[str, Any], readonly: bool = True):
        """
        Display tool call information.
        
        Args:
            tool_name: Name of the tool being called
            params: Tool parameters
            readonly: Whether the tool is readonly (affects styling)
        """
        self.start_time = time.time()
        
        # Tool header with type badge
        tool_type = "[blue]READ[/blue]" if readonly else "[magenta bold]WRITE[/magenta bold]"
        header = f"Tool: [bold]{tool_name}[/bold] {tool_type}"
        
        # Format parameters with intelligent truncation
        params_text = self._format_params(params)
        
        # Create panel
        panel = Panel(
            params_text,
            title=header,
            title_align="left",
            border_style="blue" if readonly else "magenta",
            padding=(0, 1)
        )
        
        console.print(panel)
    
    def show_result(self, success: bool, output: str, error: Optional[str] = None):
        """
        Display tool call result.
        
        Args:
            success: Whether the call succeeded
            output: Tool output/result
            error: Error message if failed
        """
        duration = time.time() - self.start_time if self.start_time else 0
        
        if success:
            status = f"[green]✓ Success[/green] [dim]({duration:.2f}s)[/dim]"
            # Truncate output to 1-2 lines preview
            preview = self._truncate_output(output)
            content = f"{status}\n{preview}" if preview else status
            border_style = "green"
        else:
            status = f"[red]✗ Failed[/red] [dim]({duration:.2f}s)[/dim]"
            error_text = self._truncate_output(error or "Unknown error")
            content = f"{status}\n[red]{error_text}[/red]"
            border_style = "red"
        
        # Result panel
        result_panel = Panel(
            content,
            title="[bold]Result[/bold]",
            title_align="left",
            border_style=border_style,
            padding=(0, 1)
        )
        console.print(result_panel)
        console.print()  # Extra line for spacing
        
        self.start_time = None
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """
        Format parameters with intelligent truncation.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Formatted parameter string
        """
        if not params:
            return "[dim]No parameters[/dim]"
        
        lines = []
        for key, value in params.items():
            # Truncate long string values
            if isinstance(value, str):
                if len(value) > 60:
                    display_value = f'"{value[:57]}..."'
                    char_count = f" [dim]({len(value)} chars)[/dim]"
                    lines.append(f"[cyan]{key}:[/cyan] {display_value}{char_count}")
                else:
                    lines.append(f"[cyan]{key}:[/cyan] \"{value}\"")
            elif isinstance(value, (list, dict)):
                # Show type and length for collections
                type_name = type(value).__name__
                length = len(value)
                lines.append(f"[cyan]{key}:[/cyan] {type_name} [dim](length: {length})[/dim]")
            else:
                lines.append(f"[cyan]{key}:[/cyan] {value}")
        
        return "\n".join(lines)
    
    def _truncate_output(self, output: str, max_lines: int = 2, max_chars: int = 150) -> str:
        """
        Truncate output to fit in preview.
        
        Args:
            output: Output text to truncate
            max_lines: Maximum number of lines
            max_chars: Maximum characters per line
            
        Returns:
            Truncated output string
        """
        if not output:
            return ""
        
        # Split into lines and take first few
        lines = output.split('\n')[:max_lines]
        
        # Truncate each line
        truncated_lines = []
        for line in lines:
            if len(line) > max_chars:
                truncated_lines.append(line[:max_chars] + "...")
            else:
                truncated_lines.append(line)
        
        result = "\n".join(truncated_lines)
        
        # Add ellipsis if we truncated lines
        total_lines = len(output.split('\n'))
        if total_lines > max_lines:
            result += f"\n[dim]... ({total_lines - max_lines} more lines)[/dim]"
        
        return result


# Global instance for convenience
tool_display = ToolCallDisplay()
