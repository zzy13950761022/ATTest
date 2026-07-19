"""
Logo and welcome screen display.
"""
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from .console import console
from .theme import STYLE_HEADER, STYLE_VALUE, STYLE_DIM

# ASCII art logo (compatible with all terminals)
LOGO = """
 _____         _      _                    _   
|_   _|__  ___| |_   / \\   __ _  ___ _ __ | |_ 
  | |/ _ \\/ __| __| / _ \\ / _` |/ _ \\ '_ \\| __|
  | |  __/\\__ \\ |_ / ___ \\ (_| |  __/ | | | |_ 
  |_|\\___||___/\\__/_/   \\_\\__, |\\___|_| |_|\\__|
                          |___/                
"""

def show_welcome(model: str, workspace: str, version: str = "0.1.0"):
    """
    Display welcome screen with logo and system information.
    
    Args:
        model: Current LLM model name
        workspace: Current workspace path
        version: Application version
    """
    console.clear()
    
    # Create logo with border
    logo_text = Text(LOGO, style="bold cyan")
    logo_panel = Panel(
        logo_text,
        title=f"[bold white]v{version}[/bold white]",
        subtitle="[dim]Python API Test Generation[/dim]",
        border_style="cyan",
        padding=(0, 2)
    )
    console.print(logo_panel)
    console.print()
    
    # System information table
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style=STYLE_HEADER, justify="right")
    info_table.add_column(style=STYLE_VALUE)
    
    info_table.add_row("Model:", model)
    info_table.add_row("Mode:", "Chat (Interactive)")
    info_table.add_row("Workspace:", workspace)
    
    info_panel = Panel(
        info_table,
        title="[bold]System Info[/bold]",
        border_style="dim",
        padding=(0, 1)
    )
    console.print(info_panel)
    console.print()
    
    # Quick start guide
    console.print("[bold cyan]Quick Start:[/bold cyan]")
    console.print("  • Just chat to use tools and get help")
    console.print("  • [bold]/workflow <target_fqn>[/bold] - Start test generation workflow")
    console.print("  • [bold]/model[/bold] - Show current model information")
    console.print("  • [bold]/help[/bold] - Show all available commands")
    console.print("  • [bold]/quit[/bold] - Exit TestAgent")
    console.print()


def show_workflow_welcome(op: str, arch: str, soc: str = "python"):
    """
    Display welcome message for workflow mode.
    
    Args:
        op: Operator name
        arch: Target architecture
        soc: SOC name
    """
    console.print()
    welcome_text = f"[bold cyan]Starting Test Generation Workflow[/bold cyan]"
    console.print(Panel(
        f"Target: [bold]{op}[/bold]\\nArchitecture: [bold]{arch}[/bold]\\nRuntime: [bold]{soc}[/bold]",
        title=welcome_text,
        border_style="cyan"
    ))
    console.print()
