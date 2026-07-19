"""
Model management functions.
"""
from typing import Dict, List

# Predefined common models
AVAILABLE_MODELS = {
    "deepseek-chat": {
        "display_name": "DeepSeek Chat",
        "base_url": "https://api.deepseek.com/v1",
        "description": "Fast and cost-effective Chinese model"
    },
    "deepseek-coder": {
        "display_name": "DeepSeek Coder",
        "base_url": "https://api.deepseek.com/v1",
        "description": "Specialized for code generation"
    },
    "gpt-4o": {
        "display_name": "GPT-4o",
        "base_url": "https://api.openai.com/v1",
        "description": "Latest OpenAI multimodal model"
    },
    "gpt-4o-mini": {
        "display_name": "GPT-4o Mini",
        "base_url": "https://api.openai.com/v1",
        "description": "Faster and cheaper GPT-4o variant"
    },
    "claude-3-5-sonnet": {
        "display_name": "Claude 3.5 Sonnet",
        "base_url": "https://api.anthropic.com/v1",
        "description": "Anthropic's most capable model"
    }
}


def list_available_models(use_rich: bool = False):
    """List all predefined models."""
    if use_rich:
        from testagent_cli.ui.console import console
        from rich.table import Table
        from rich.panel import Panel
        
        model_table = Table(show_header=True, header_style="bold cyan")
        model_table.add_column("Model ID", style="cyan")
        model_table.add_column("Display Name", style="bold")
        model_table.add_column("Description", style="dim")
        
        for model_id, info in AVAILABLE_MODELS.items():
            model_table.add_row(
                model_id,
                info["display_name"],
                info["description"]
            )
        
        console.print()
        console.print(Panel(
            model_table,
            title="[bold]Available Models[/bold]",
            subtitle="[dim]Use /model switch <model_id> to switch[/dim]",
            border_style="cyan"
        ))
        console.print()
    else:
        print("\n=== Available Models ===")
        for model_id, info in AVAILABLE_MODELS.items():
            print(f"  {model_id:20s} - {info['display_name']:20s} {info['description']}")
        print("\nUse /model switch <model_id> to switch")
        print()


def switch_model(llm, model_name: str, use_rich: bool = False):
    """
    Switch to a different model (temporary, for current session only).
    
    Args:
        llm: LLM client instance to update
        model_name: Name of the model to switch to
        use_rich: Whether to use rich formatting
    """
    if model_name not in AVAILABLE_MODELS:
        if use_rich:
            from testagent_cli.ui.console import console
            console.print(f"[red]Error:[/red] Unknown model '{model_name}'")
            console.print("[yellow]Hint:[/yellow] Use /model list to see available models")
        else:
            print(f"Error: Unknown model '{model_name}'")
            print("Hint: Use /model list to see available models")
        return
    
    model_info = AVAILABLE_MODELS[model_name]
    
    # Update LLM client (temporary, session-only)
    llm.model = model_name
    llm.base_url = model_info["base_url"]
    
    if use_rich:
        from testagent_cli.ui.console import console
        from rich.panel import Panel
        console.print()
        console.print(Panel(
            f"[green]✓[/green] Switched to [bold]{model_info['display_name']}[/bold] ({model_name})\n"
            f"[dim]Note: This change is temporary for the current session only[/dim]",
            title="[bold]Model Switched[/bold]",
            border_style="green"
        ))
        console.print()
    else:
        print(f"\n✓ Switched to {model_info['display_name']} ({model_name})")
        print("Note: This change is temporary for the current session only\n")
