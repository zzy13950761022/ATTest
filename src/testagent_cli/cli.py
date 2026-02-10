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
config_app = typer.Typer(help="配置管理")
sessions_app = typer.Typer(help="会话管理")
app.add_typer(config_app, name="config")
app.add_typer(sessions_app, name="sessions")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: Optional[Path] = typer.Option(None, help="工作目录"),
    auto_approve: bool = typer.Option(False, help="自动批准写/执行"),
):
    """
    ATTest CLI - Python/PyTorch/TensorFlow 测试用例生成工具

    默认启动交互式聊天模式，可使用 /workflow 命令进入测试生成工作流
    """
    if ctx.invoked_subcommand is None:
        # No subcommand specified, enter interactive chat mode
        workspace_path = workspace.expanduser().resolve() if workspace else Path.cwd()
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Start interactive mode with welcome screen
        run_interactive_mode(str(workspace_path), auto_approve)


def run_interactive_mode(workspace: str, auto_approve: bool = False):
    """
    启动交互式模式，显示欢迎界面并进入chat循环
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
    typer.echo(f"已设置 {key} = {parsed}")


@config_app.command("get")
def config_get(key: str):
    val = get_config_value(key)
    if val is None:
        typer.echo("未找到")
    else:
        typer.echo(val)


@config_app.command("list")
def config_list():
    typer.echo(json.dumps(list_config(), indent=2, ensure_ascii=False))
    typer.echo(f"\n配置文件: {CONFIG_PATH}")


@sessions_app.command("list")
def sessions_list():
    sessions = list_sessions()
    if not sessions:
        typer.echo("无会话")
    else:
        for s in sessions:
            typer.echo(s)


@sessions_app.command("clear")
def sessions_clear(session_id: str):
    clear_session(session_id)
    typer.echo(f"已清除 {session_id}")


@app.command("chat")
def run_chat_mode(
    workspace: Path = typer.Option(".", help="工作目录"),
    auto_approve: bool = typer.Option(False, help="自动批准写/执行"),
):
    """启动交互式聊天模式（遗留命令，推荐直接使用 testagent）"""
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
    )
    engine.run(mode=mode)


@app.command("run")
def run_python(
    func: str = typer.Option(..., "-f", "--func", help="目标函数全限定名，如 package.module:func 或 package.module.Class.method"),
    workspace: Path = typer.Option(".", "-w", "--workspace", help="工作目录（将自动加入 PYTHONPATH）"),
    project_root: Optional[Path] = typer.Option(None, "-p", "--project-root", help="项目根目录，默认与 workspace 相同"),
    mode: str = typer.Option("interactive", "-m", "--mode", help="模式: interactive | full-auto"),
    resume: bool = typer.Option(False, "-r", "--resume", help="恢复上次中断的工作流"),
    epoch: int = typer.Option(1, "-e", "--epoch", help="全自动模式下的迭代轮数：在 analyze_results 后回到 generate_code 迭代指定轮数，再生成报告"),
):
    """针对 Python 函数生成并运行测试用例"""
    _launch_python_workflow(func, workspace, project_root, mode, resume, epoch)


@app.command("test")
def run_testagent(
    func: str = typer.Option(..., "-f", "--func", help="兼容命令，填入目标函数全限定名，如 torch.nn.functional.relu"),
    workspace: Path = typer.Option(".", "-w", "--workspace", help="工作目录（默认作为项目根目录）"),
    project_root: Optional[Path] = typer.Option(None, "-p", "--project-root", help="项目根目录"),
    mode: str = typer.Option("interactive", "-m", "--mode", help="模式: interactive | full-auto"),
    resume: bool = typer.Option(False, "-r", "--resume", help="恢复上次中断的工作流"),
    epoch: int = typer.Option(1, "-e", "--epoch", help="全自动模式迭代轮数：分析后回到生成代码重复，默认1"),
):
    """运行测试生成工作流（兼容旧命令）"""
    _launch_python_workflow(func, workspace, project_root, mode, resume, epoch)


if __name__ == "__main__":
    app()
