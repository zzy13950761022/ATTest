"""
Progress visualization and user interaction for workflow.
"""
from typing import List
from .state import WorkflowState

# Try to import rich components
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.text import Text
    from rich.markup import escape
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False


def show_progress(state: WorkflowState, stages: List[str]):
    """
    Display workflow progress with visual indicators.
    """
    if RICH_AVAILABLE:
        _show_progress_rich(state, stages)
    else:
        _show_progress_plain(state, stages)


def _show_progress_rich(state: WorkflowState, stages: List[str]):
    """Rich-formatted progress display."""
    completed_stages = {r.stage for r in state.stage_history if r.status == "completed"}
    
    # Create progress tree
    target = getattr(state, "target", state.op)
    tree = Tree(f"[bold cyan]ATTest Workflow[/bold cyan] - {target} ({state.arch})")
    
    for i, stage_name in enumerate(stages):
        # Determine status
        if stage_name == state.current_stage:
            icon = "▶"
            style = "bold yellow"
            suffix = " ← Current"
        elif stage_name in completed_stages:
            icon = "✓"
            style = "green"
            suffix = ""
        else:
            icon = "○"
            style = "dim"
            suffix = ""
        
        # Get display name
        display = stage_name.replace("_", " ").title()
        label = f"[{style}]{icon} {i+1}. {display}{suffix}[/{style}]"
        tree.add(label)
    
    console.print()
    console.print(tree)
    console.print()


def _show_progress_plain(state: WorkflowState, stages: List[str]):
    """Fallback plain text progress display."""
    width = 61
    print("╔" + "═" * width + "╗")
    target = getattr(state, "target", state.op)
    title = f"ATTest Workflow - {target} ({state.arch})"
    padding = (width - len(title)) // 2
    print(f"║{' ' * padding}{title}{' ' * (width - padding - len(title))}║")
    print("╠" + "═" * width + "╣")
    
    completed_stages = {r.stage for r in state.stage_history if r.status == "completed"}
    
    for i, stage_name in enumerate(stages):
        if stage_name == state.current_stage:
            symbol = "▶"
            suffix = " ← Current Stage"
        elif stage_name in completed_stages:
            symbol = "●"
            suffix = ""
        else:
            symbol = " "
            suffix = ""
        
        display = stage_name.replace("_", " ").title()
        line = f" [{symbol}] {i+1}. {display}"
        remaining_space = width - len(line) - len(suffix)
        line = f"║{line}{' ' * remaining_space}{suffix}║"
        print(line)
    
    print("╚" + "═" * width + "╝")


def show_stage_output(stage_name: str, result_message: str, artifacts: List[str]):
    """Display stage completion with artifact list."""
    if RICH_AVAILABLE:
        _show_stage_output_rich(stage_name, result_message, artifacts)
    else:
        _show_stage_output_plain(stage_name, result_message, artifacts)


def _show_stage_output_rich(stage_name: str, result_message: str, artifacts: List[str]):
    """Rich-formatted stage output."""
    display_name = stage_name.replace("_", " ").title()
    
    # Create content
    safe_msg = escape(result_message or "Stage completed successfully")
    content = Text()
    content.append("✓ ", style="green")
    content.append(safe_msg)
    if artifacts:
        content.append("\n\nGenerated artifacts:", style="bold")
        for artifact in artifacts:
            content.append(f"\n  • {artifact}")
    
    console.print()
    console.print(Panel(
        content,
        title=f"[bold]{display_name}[/bold]",
        border_style="green",
        padding=(0, 2)
    ))


def _show_stage_output_plain(stage_name: str, result_message: str, artifacts: List[str]):
    """Fallback plain text stage output."""
    print(f"\n✓ Stage '{stage_name}' completed - {result_message or 'Stage completed successfully'}")
    if artifacts:
        print(f"  📄 Generated: {', '.join(artifacts)}")


def collect_feedback(allow_commands: bool = True) -> str:
    """
    Collect user feedback with command support.
    
    Returns:
        User input string
    """
    if RICH_AVAILABLE:
        return _collect_feedback_rich(allow_commands)
    else:
        return _collect_feedback_plain(allow_commands)


def _collect_feedback_rich(allow_commands: bool) -> str:
    """Rich-formatted feedback collection."""
    console.print()
    
    # Show command hints
    if allow_commands:
        hints_text = "[dim]Commands: /next | /regenerate | /retry <msg> | /goto <stage> | /status | /quit[/dim]"
        console.print(hints_text)
    
    try:
        user_input = console.input("[bold cyan]>[/bold cyan] ").strip()
        return user_input
    except (EOFError, KeyboardInterrupt):
        return "/quit"


def _collect_feedback_plain(allow_commands: bool) -> str:
    """Fallback plain text feedback collection."""
    print("-" * 61)
    if allow_commands:
        print("Commands: /next | /regenerate | /retry <msg> | /goto <stage> | /status | /quit")
    
    try:
        user_input = input("> ").strip()
        return user_input
    except (EOFError, KeyboardInterrupt):
        return "/quit"


def show_artifact_content(artifact_name: str, content: str, max_lines: int = 20):
    """Display artifact content with truncation."""
    if RICH_AVAILABLE:
        _show_artifact_content_rich(artifact_name, content, max_lines)
    else:
        _show_artifact_content_plain(artifact_name, content, max_lines)


def _show_artifact_content_rich(artifact_name: str, content: str, max_lines: int):
    """Rich-formatted artifact content."""
    lines = content.split('\n')
    if len(lines) > max_lines:
        display_content = '\n'.join(lines[:max_lines])
        display_content += f"\n\n[dim]... ({len(lines) - max_lines} more lines)[/dim]"
    else:
        display_content = content
    
    console.print()
    console.print(Panel(
        display_content,
        title=f"[bold]📄 {artifact_name}[/bold]",
        border_style="cyan",
        padding=(0, 2)
    ))


def _show_artifact_content_plain(artifact_name: str, content: str, max_lines: int):
    """Fallback plain text artifact content."""
    print(f"\n📄 {artifact_name}:")
    print("─" * 61)
    
    lines = content.split('\n')
    if len(lines) > max_lines:
        for line in lines[:max_lines]:
            print(line)
        print(f"... ({len(lines) - max_lines} more lines)")
    else:
        print(content)
    
    print("─" * 61)


def show_error(error_message: str):
    """Display error message."""
    if RICH_AVAILABLE:
        console.print(f"\n[red]✗ Error:[/red] {error_message}\n")
    else:
        print(f"\n✗ Error: {error_message}")


def show_help():
    """Show available commands."""
    if RICH_AVAILABLE:
        _show_help_rich()
    else:
        _show_help_plain()


def _show_help_rich():
    """Rich-formatted help."""
    help_table = Table(show_header=True, header_style="bold cyan")
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description")
    
    help_table.add_row("/next", "Approve and continue to next stage")
    help_table.add_row("/regenerate", "Retry current stage")
    help_table.add_row("/retry <msg>", "Retry with additional feedback")
    help_table.add_row("/goto <stage>", "Jump to specific stage")
    help_table.add_row("/status", "Show workflow state")
    help_table.add_row("/quit", "Exit workflow")
    
    console.print()
    console.print(Panel(
        help_table,
        title="[bold]Available Commands[/bold]",
        subtitle="[dim]Or provide natural language feedback[/dim]",
        border_style="cyan"
    ))
    console.print()


def _show_help_plain():
    """Fallback plain text help."""
    print("\nAvailable commands:")
    print("  /next          - Approve and continue to next stage")
    print("  /regenerate    - Retry current stage")
    print("  /retry <msg>   - Retry with additional feedback")
    print("  /goto <stage>  - Jump to specific stage")
    print("  /status        - Show workflow state")
    print("  /quit          - Exit workflow")
    print("\nOr provide natural language feedback to guide the agent.")
