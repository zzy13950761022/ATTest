import json
import os
from pathlib import Path
from typing import Optional

import typer

from .config import (
    CONFIG_PATH,
    list_config,
    load_config,
    save_config,
    set_config_value,
    get_config_value,
)
from .llm import LLMClient
from .session import append_message, clear_session, list_sessions, load_history
from .chat import run_chat
from .utils import slugify_target


app = typer.Typer(help="ATTest CLI - Python API test generation assistant")
config_app = typer.Typer(help="Configuration Management")
sessions_app = typer.Typer(help="Session Management")
ascend_ut_app = typer.Typer(help="Ascend C operator UT workflow")
app.add_typer(config_app, name="config")
app.add_typer(sessions_app, name="sessions")
app.add_typer(ascend_ut_app, name="ascend-ut")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: Optional[Path] = typer.Option(None, help="Workspace"),
    auto_approve: bool = typer.Option(False, help="Auto-approve write/exec"),
):
    """
    ATTest CLI - Test generation tool for Python/PyTorch/TensorFlow

    Starts in interactive chat mode by default; use `/workflow` to enter the test generation workflow
    """
    if ctx.invoked_subcommand is None:
        # No subcommand specified, enter interactive chat mode
        workspace_path = workspace.expanduser().resolve() if workspace else Path.cwd()
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Start interactive mode with welcome screen
        run_interactive_mode(str(workspace_path), auto_approve)


def run_interactive_mode(workspace: str, auto_approve: bool = False):
    """
    Start interactive mode, show the welcome screen, and enter the chat loop
    """
    cfg = load_config()
    
    # Get model info
    model_name = cfg.get("api", {}).get("model", "deepseek-chat")
    
    # Show welcome screen
    try:
        from .ui.logo import show_welcome
        show_welcome(model=model_name, workspace=workspace)
    except ImportError:
        # Fallback if rich is not installed
        print("=" * 60)
        print("ATTest CLI - Intelligent Test Generation")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Workspace: {workspace}")
        print()
    
    # Enter chat mode
    run_chat(workspace, auto_approve)


@config_app.command("set")
def config_set(key: str, value: str):
    parsed: str | int | float | bool
    if value.lower() in {"true", "false"}:
        parsed = value.lower() == "true"
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
    set_config_value(key, parsed)
    typer.echo(f"Set {key} = {parsed}")


@config_app.command("get")
def config_get(key: str):
    val = get_config_value(key)
    if val is None:
        typer.echo("Not found")
    else:
        typer.echo(val)


@config_app.command("list")
def config_list():
    typer.echo(json.dumps(list_config(), indent=2, ensure_ascii=False))
    typer.echo(f"\nConfig file: {CONFIG_PATH}")


@sessions_app.command("list")
def sessions_list():
    sessions = list_sessions()
    if not sessions:
        typer.echo("No sessions")
    else:
        for s in sessions:
            typer.echo(s)


@sessions_app.command("clear")
def sessions_clear(session_id: str):
    clear_session(session_id)
    typer.echo(f"Cleared {session_id}")


@app.command("chat")
def run_chat_mode(
    workspace: Path = typer.Option(".", help="Workspace"),
    auto_approve: bool = typer.Option(False, help="Auto-approve write/exec"),
):
    """Start interactive chat mode (legacy command; using `attest` directly is recommended)"""
    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    run_interactive_mode(str(workspace), auto_approve)


def _launch_python_workflow(
    func: str,
    workspace: Path,
    project_root: Optional[Path],
    mode: str,
    resume: bool,
    epochs: int = 1,
):
    """Shared launcher for Python workflow commands."""
    from .workflow import WorkflowEngine

    cfg = load_config()

    workspace = workspace.expanduser().resolve()
    if project_root is None:
        project_root = workspace
    else:
        project_root = project_root.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    llm = LLMClient(
        model=cfg["api"].get("model", "deepseek-chat"),
        base_url=cfg["api"].get("base_url", ""),
        api_key=cfg["api"].get("api_key", ""),
    )

    target_slug = slugify_target(func)
    engine = WorkflowEngine(
        llm=llm,
        workspace=str(workspace),
        op=target_slug,
        arch="python",
        soc="python",
        vendor="python",
        project_root=str(project_root),
        target=func,
        resume=resume,
        epochs=epochs,
        workflow_kind="python",
    )
    engine.run(mode=mode)


def _launch_ascend_workflow(
    op_path: str,
    workspace: Path,
    project_root: Path,
    soc: Optional[str],
    skill_root: Optional[Path],
    mode: str,
    resume: bool,
    epochs: int,
    profile: str = "ascend_ut",
):
    from .workflow import WorkflowEngine
    from .ascend import DEFAULT_ASCEND_SKILL_ROOT, normalize_operator_path

    cfg = load_config()
    ascend_cfg = cfg.get("profiles", {}).get("ascend_ut", {})

    workspace = workspace.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    selected_soc = soc or ascend_cfg.get("default_soc", "Ascend910B")
    selected_skill_root = (
        str(skill_root.expanduser().resolve())
        if skill_root is not None
        else ascend_cfg.get("skill_root", DEFAULT_ASCEND_SKILL_ROOT)
    )
    normalized = normalize_operator_path(project_root, op_path)

    llm = LLMClient(
        model=cfg["api"].get("model", "deepseek-chat"),
        base_url=cfg["api"].get("base_url", ""),
        api_key=cfg["api"].get("api_key", ""),
    )

    engine = WorkflowEngine(
        llm=llm,
        workspace=str(workspace),
        op=normalized["op_name"],
        arch="ascendc",
        soc=selected_soc,
        vendor="ascend",
        project_root=str(project_root),
        target=normalized["canonical_op_path"],
        resume=resume,
        epochs=epochs,
        workflow_kind=profile,
        op_path=normalized["canonical_op_path"],
        repo_name=normalized["repo_name"],
        category=normalized["category"],
        op_name=normalized["op_name"],
        generation_mode="ut_generate",
        coverage_mode="single_run",
        enabled_layers=[],
        skill_root=selected_skill_root,
    )
    engine.run(mode=mode)


@app.command("run")
def run_python(
    func: str = typer.Option(..., "-f", "--func", help="Fully qualified target function name, for example `package.module:func` or `package.module.Class.method`"),
    workspace: Path = typer.Option(".", "-w", "--workspace", help="Workspace (automatically added to PYTHONPATH)"),
    project_root: Optional[Path] = typer.Option(None, "-p", "--project-root", help="Project root directory; defaults to the same value as `workspace`"),
    mode: str = typer.Option("interactive", "-m", "--mode", help="Mode: `interactive` | `full-auto`"),
    resume: bool = typer.Option(False, "-r", "--resume", help="Resume the previously interrupted workflow"),
    epoch: int = typer.Option(1, "-e", "--epoch", help="Iteration count in full-auto mode: after `analyze_results`, return to `generate_code` for the specified number of rounds, then generate the report"),
):
    """Generate and run test cases for a Python function"""
    _launch_python_workflow(func, workspace, project_root, mode, resume, epoch)


@app.command("test")
def run_attest(
    func: str = typer.Option(..., "-f", "--func", help="Compatibility command. Provide the fully qualified target function name, such as `torch.nn.functional.relu`"),
    workspace: Path = typer.Option(".", "-w", "--workspace", help="Workspace (used as the project root by default)"),
    project_root: Optional[Path] = typer.Option(None, "-p", "--project-root", help="Project root directory"),
    mode: str = typer.Option("interactive", "-m", "--mode", help="Mode: `interactive` | `full-auto`"),
    resume: bool = typer.Option(False, "-r", "--resume", help="Resume the previously interrupted workflow"),
    epoch: int = typer.Option(1, "-e", "--epoch", help="Iteration count in full-auto mode: after analysis, return to code generation; defaults to 1"),
):
    """Run the test generation workflow (legacy-compatible command)"""
    _launch_python_workflow(func, workspace, project_root, mode, resume, epoch)


@ascend_ut_app.command("run")
def run_ascend_ut(
    op_path: str = typer.Option(..., "--op-path", help="Operator path: <category>/<op> or <repo>/<category>/<op>"),
    workspace: Path = typer.Option(".", "-w", "--workspace", help="Workspace for .attest state and artifacts"),
    project_root: Path = typer.Option(..., "-p", "--project-root", help="Ascend operator repository root containing build.sh"),
    soc: Optional[str] = typer.Option(None, "--soc", help="Preferred SoC, such as Ascend910B"),
    skill_root: Optional[Path] = typer.Option(None, "--skill-root", help="Override Ascend skill root"),
    mode: str = typer.Option("interactive", "-m", "--mode", help="Mode: `interactive` | `full-auto`"),
    resume: bool = typer.Option(False, "-r", "--resume", help="Resume the previously interrupted workflow"),
    epoch: int = typer.Option(1, "-e", "--epoch", help="Iteration count in full-auto mode"),
    profile: str = typer.Option("ascend_ut", "--profile", help="Workflow profile: ascend_ut | ascend_ut_continuous"),
):
    """Generate and run Ascend C operator UT in the dedicated Ascend workflow."""
    _launch_ascend_workflow(op_path, workspace, project_root, soc, skill_root, mode, resume, epoch, profile)


if __name__ == "__main__":
    app()
