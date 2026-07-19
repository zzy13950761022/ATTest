"""
Slash command handlers for chat mode.
"""
from typing import Optional
from .config import load_config
from .models import list_available_models, switch_model


def handle_slash_command(command: str, workspace: str, llm) -> bool:
    """
    Handle slash commands in chat mode.
    
    Args:
        command: The command string (starting with /)
        workspace: Current workspace path
        llm: LLM client instance
        
    Returns:
        True if should exit chat, False otherwise
    """
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    try:
        from .ui.console import console
        from rich.panel import Panel
        from rich.table import Table
        use_rich = True
    except ImportError:
        use_rich = False
    
    # /quit or /exit
    if cmd in ["/quit", "/exit"]:
        if use_rich:
            console.print("[cyan]Goodbye! 👋[/cyan]")
        else:
            print("Goodbye!")
        return True
    
    # /help
    elif cmd == "/help":
        show_help(use_rich)
        return False
    
    # /model [subcommand]
    elif cmd == "/model":
        # Parse model subcommand
        if not args:
            show_model_info(llm, use_rich)
        elif args.startswith("list"):
            list_available_models(use_rich)
        elif args.startswith("switch "):
            model_name = args[7:].strip()  # Remove "switch "
            switch_model(llm, model_name, use_rich)
        else:
            if use_rich:
                console.print(f"[red]Unknown /model subcommand:[/red] {args}")
                console.print("[yellow]Usage:[/yellow] /model [list|switch <name>]")
            else:
                print(f"Unknown /model subcommand: {args}")
                print("Usage: /model [list|switch <name>]")
        return False
    
    # /workflow [target_fqn]
    elif cmd == "/workflow":
        # Check if we have args, otherwise enter interactive mode
        if not args:
            # Interactive mode - prompt for parameters
            start_workflow_interactive(workspace, use_rich)
        else:
            target_fqn = args.strip()
            start_workflow(target_fqn, workspace, use_rich)
        return False
    
    else:
        if use_rich:
            console.print(f"[red]Unknown command:[/red] {cmd}")
            console.print("[dim]Type /help to see available commands[/dim]")
        else:
            print(f"Unknown command: {cmd}")
            print("Type /help to see available commands")
        return False


def show_help(use_rich: bool = False):
    """Display help information."""
    if use_rich:
        from .ui.console import console
        from rich.table import Table
        from rich.panel import Panel
        
        help_table = Table(show_header=True, header_style="bold cyan")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description")
        help_table.add_column("Example", style="dim")
        
        help_table.add_row(
            "/workflow <target_fqn>",
            "Start Python test generation workflow",
            "/workflow torch.nn.functional.relu"
        )
        help_table.add_row(
            "/model [list|switch <name>]",
            "Manage models",
            "/model list"
        )
        help_table.add_row(
            "/help",
            "Show this help message",
            "/help"
        )
        help_table.add_row(
            "/quit",
            "Exit TestAgent",
            "/quit"
        )
        
        console.print()
        console.print(Panel(help_table, title="[bold]Available Commands[/bold]", border_style="cyan"))
        console.print()
    else:
        print("\n=== Available Commands ===")
        print("  /workflow <target_fqn>      - Start test generation workflow")
        print("  /model                       - Show current model information")
        print("  /help                        - Show this help message")
        print("  /quit                        - Exit TestAgent")
        print()


def show_model_info(llm, use_rich: bool = False):
    """Display current model information."""
    cfg = load_config()
    model_name = cfg.get("api", {}).get("model", "unknown")
    base_url = cfg.get("api", {}).get("base_url", "")
    
    if use_rich:
        from .ui.console import console
        from rich.panel import Panel
        from rich.table import Table
        
        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style="bold cyan", justify="right")
        info_table.add_column(style="green")
        
        info_table.add_row("Model:", model_name)
        info_table.add_row("Base URL:", base_url or "[dim]default[/dim]")
        info_table.add_row("Temperature:", str(cfg.get("api", {}).get("temperature", 0.2)))
        
        console.print()
        console.print(Panel(info_table, title="[bold]Model Information[/bold]", border_style="cyan"))
        console.print()
    else:
        print("\n=== Model Information ===")
        print(f"Model: {model_name}")
        print(f"Base URL: {base_url or 'default'}")
        print(f"Temperature: {cfg.get('api', {}).get('temperature', 0.2)}")
        print()


def start_workflow(target_fqn: str, workspace: str, use_rich: bool = False):
    """Start workflow mode for a Python target."""
    from .workflow import WorkflowEngine
    from .llm import LLMClient
    from .config import load_config
    from .utils import slugify_target
    
    target_slug = slugify_target(target_fqn)
    
    if use_rich:
        from .ui.console import console
        from .ui.logo import show_workflow_welcome
        console.print()
        show_workflow_welcome(target_fqn, arch="python", soc="python")
    else:
        print(f"\nStarting workflow for target '{target_fqn}'...")
    
    cfg = load_config()
    
    llm = LLMClient(
        model=cfg["api"].get("model", "deepseek-chat"),
        base_url=cfg["api"].get("base_url", ""),
        api_key=cfg["api"].get("api_key", ""),
    )
    
    engine = WorkflowEngine(
        llm=llm,
        workspace=workspace,
        op=target_slug,
        arch="python",
        soc="python",
        vendor="python",
        project_root=workspace,
        target=target_fqn,
        resume=False
    )
    
    try:
        engine.run(mode="interactive")
    except KeyboardInterrupt:
        if use_rich:
            from .ui.console import console
            console.print("\n[yellow]Workflow interrupted. Returning to chat mode...[/yellow]")
        else:
            print("\nWorkflow interrupted. Returning to chat mode...")
    except Exception as e:
        if use_rich:
            from .ui.console import console
            console.print(f"\n[red]Error:[/red] {e}")
        else:
            print(f"\nError: {e}")


def start_workflow_interactive(workspace: str, use_rich: bool = False):
    """
    Start workflow in interactive mode - prompt for parameters.
    """
    cfg = load_config()

    if use_rich:
        from testagent_cli.ui.console import console
        from rich.panel import Panel
        
        console.print()
        console.print(Panel(
            "[bold cyan]Test Generation Workflow[/bold cyan]\n\n"
            "This wizard will guide you through setting up a Python test generation workflow.",
            border_style="cyan"
        ))
        console.print()
        
        console.print("[bold]Which Python API do you want to test?[/bold]")
        console.print("[dim]Examples: torch.nn.functional.relu, torch.add, tensorflow.math.add[/dim]")
        target = console.input("[cyan]Target FQN:[/cyan] ").strip()
        
        if not target:
            console.print("[red]Error:[/red] Target is required")
            return
    else:
        print("\n=== Test Generation Workflow ===")
        print("This wizard will guide you through setting up a Python test generation workflow.\n")
        
        target = input("Target function (FQN): ").strip()
        if not target:
            print("Error: Target is required")
            return
    
    # Start workflow with collected parameters
    start_workflow(target, workspace, use_rich)
