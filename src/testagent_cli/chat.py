"""
Chat mode for interactive REPL with LLM and tool calling.
"""
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

from .llm import LLMClient, ChatResponse
from .tools import ToolRegistry, ToolRunner, ToolContext, build_default_registry
from .config import load_config
from .session import append_message, load_history
from .commands import handle_slash_command


def run_chat(workspace: str, auto_approve: bool = False):
    """
    Run interactive chat mode with LLM and tool access.
    
    Args:
        workspace: Working directory for tool operations
        auto_approve: If True, skip approval prompts for mutating tools
    """
    # Load config
    cfg = load_config()
    api_cfg = cfg.get("api", {})
    
    # Initialize LLM client
    llm = LLMClient(
        base_url=api_cfg.get("base_url", ""),
        api_key=api_cfg.get("api_key", ""),
        model=api_cfg.get("model", "deepseek-chat"),
        temperature=api_cfg.get("temperature", 0.2),
        max_tokens=api_cfg.get("max_tokens", 4096)
    )
    
    # Initialize tools
    registry = build_default_registry()
    runner = ToolRunner(registry)
    tool_schemas = registry.to_llm_schema()
    
    # Session management
    session_id = str(uuid.uuid4())[:8]
    messages: List[Dict[str, Any]] = []
    
    # System prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful coding assistant with access to file system tools. "
            "Use tools when necessary to help the user. "
            "Always explain what you're doing when calling tools."
        )
    }
    messages.append(system_prompt)
    
    print(f"[Chat Mode] Session: {session_id}")
    print(f"[Chat Mode] Workspace: {workspace}")
    print(f"[Chat Mode] Type 'exit' or Ctrl+C to quit\n")
    
    # REPL loop
    while True:
        try:
            # Get user input with rich prompt
            try:
                from .ui.console import console
                user_input = console.input("[bold cyan]>[/bold cyan] ").strip()
                use_rich = True
            except ImportError:
                user_input = input("> ").strip()
                use_rich = False
            
            if not user_input:
                continue
            
            # Handle slash commands
            if user_input.startswith("/"):
                if handle_slash_command(user_input, workspace, llm):
                    # Command requested exit
                    break
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                try:
                    from .ui.console import console
                    console.print("[cyan]Goodbye! 👋[/cyan]")
                except ImportError:
                    print("[Chat Mode] Goodbye!")
                break
            
            # Add user message
            user_msg = {"role": "user", "content": user_input}
            messages.append(user_msg)
            append_message(session_id, "user", user_input, workspace=workspace)
            
            # Multi-turn tool calling loop
            max_iterations = 100
            for iteration in range(max_iterations):
                # Call LLM with loading animation
                try:
                    if use_rich:
                        from rich.spinner import Spinner
                        from rich.live import Live
                        spinner = Spinner("dots", text="[dim]Thinking...[/dim]")
                        with Live(spinner, console=console, transient=True):
                            response = llm.chat(messages, tools=tool_schemas)
                    else:
                        response = llm.chat(messages, tools=tool_schemas)
                except Exception as e:
                    if use_rich:
                        console.print(f"[red]✗ LLM API error:[/red] {e}\n")
                    else:
                        print(f"[Error] LLM API error: {e}")
                    break
                
                # Add assistant response to messages
            assistant_msg = {
                "role": "assistant",
                "content": response.content,
                "reasoning_content": getattr(response, "reasoning_content", "") or ""
            }
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)
            append_message(session_id, "assistant", assistant_msg, workspace=workspace)
            
            # If no tool calls, show response and break
            if not response.has_tool_calls():
                if response.content:
                    if use_rich:
                        from rich.panel import Panel
                        console.print()
                        console.print(Panel(
                            response.content,
                            title="[bold green]ATTest[/bold green]",
                            border_style="green",
                            padding=(0, 1)
                        ))
                        console.print()
                    else:
                        print(f"\nATTest: {response.content}\n")
                    append_message(session_id, "assistant", response.content, workspace=workspace)
                    break
                
                # Execute tool calls (tool_display already handles the UI)
                ctx = ToolContext(cwd=workspace, auto_approve=auto_approve)
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    # Execute tool (tool_display handles all output)
                    result = runner.execute(tool_name, tool_args, ctx)
                    
                    # Add tool result to messages
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result.output if result.ok else f"Error: {result.error}"
                    }
                    messages.append(tool_msg)
                    append_message(
                        session_id,
                        "tool",
                        tool_msg["content"],
                        workspace=workspace,
                    )
                
                # Continue loop to let LLM process tool results
            
            if iteration >= max_iterations - 1:
                if use_rich:
                    console.print("[yellow]⚠ Max tool calling iterations reached[/yellow]\n")
                else:
                    print("[Warning] Max tool calling iterations reached")
                
        except KeyboardInterrupt:
            print("\n[Chat Mode] Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"[Error] Unexpected error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Entry point for testing."""
    import sys
    workspace = sys.argv[1] if len(sys.argv) > 1 else "."
    run_chat(workspace)


if __name__ == "__main__":
    main()
