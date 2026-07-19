"""
Stage 5: Execute Tests
Runs build and test commands, capturing logs and results.
"""
import json
from pathlib import Path
from ..stage import Stage, StageConfig, StageResult
from ..state import WorkflowState
from ...tools import ToolContext


class ExecutionStage(Stage):
    """
    Execute build and run scripts, capturing all output and exit codes.
    """
    
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="execute_tests",
            display_name="Execute Tests",
            description="Build and run test cases",
            prompt_template="",  # Not used, custom execution
            input_artifacts=[],  # No artifacts needed, files are in project_root
            output_artifacts=["execution_log.txt", "exit_code.txt"],
            tools=["exec_command"],
            allow_skip=False
        )
    
    def execute(self, state: WorkflowState) -> StageResult:
        """
        Custom execution logic for Python/pytest workflow.
        """
        # Load configuration
        from ...config import load_config
        config = load_config()
        commands = config.get("commands", {})
        project_cfg = config.get("project", {})
        test_template = project_cfg.get("test_file_template", "")
        if test_template:
            test_file_path = test_template.format(
                op=state.op,
                target=state.target,
                target_slug=state.target_slug,
            )
        else:
            test_file_path = f"tests/test_{state.target_slug}.py"

        test_file_path = self._resolve_test_file_path(state, test_file_path)
        
        # Validate config
        if not commands:
            return StageResult(
                success=False,
                outputs={},
                error="No commands configured. Please set up project.commands in config.json"
            )
        
        # Use project_root as working directory
        project_root = Path(state.project_root)
        if not project_root.exists():
            return StageResult(
                success=False,
                outputs={},
                error=f"Project root does not exist: {project_root}"
            )
        
        ctx = ToolContext(cwd=str(project_root), auto_approve=True)
        outputs = {}
        full_log = []
        
        # Step 1: Compile (optional for Python)
        compile_cmd_template = commands.get("compile", "") or ""
        if compile_cmd_template.strip():
            compile_cmd = compile_cmd_template.format(
                op=state.op,
                soc=state.soc,
                arch=state.arch,
                vendor=state.vendor,
                project_root=str(state.project_root),
                test_file_path=test_file_path,
            )
            
            print(f"  🔨 Step 1: Compiling (optional)...")
            print(f"     Command: {compile_cmd}")
            print(f"     Working dir: {project_root}")
            
            compile_result = self.tool_runner.execute(
                "exec_command",
                {"cmd": compile_cmd},
                ctx
            )
            
            compile_log = compile_result.output if compile_result.ok else (
                compile_result.output + f"\nError: {compile_result.error}"
            )
            full_log.append(f"=== Compile ===\n{compile_log}")
            
            if not compile_result.ok:
                final_log = "\n\n".join(full_log)
                outputs["execution_log.txt"] = final_log
                state.save_artifact("execution_log.txt", final_log)
                return StageResult(
                    success=False,
                    outputs=outputs,
                    error="Compile failed",
                    message=f"Compilation failed:\n{compile_log[:300]}"
                )
            
            print("     ✓ Compile successful")
        else:
            print("  ⏭️  Step 1: No compile command configured, skipping")
        
        # Step 2: Install (optional)
        install_cmd_template = commands.get("install", "") or ""
        if install_cmd_template.strip():
            output_binary = config.get("project", {}).get("output_binary_template", "").format(
                op=state.op,
                target=state.target,
                target_slug=state.target_slug,
            )
            install_cmd = install_cmd_template.format(
                op=state.op,
                soc=state.soc,
                output_binary=output_binary,
                project_root=str(state.project_root),
                test_file_path=test_file_path,
            )
            
            print(f"  📦 Step 2: Installing...")
            
            install_result = self.tool_runner.execute(
                "exec_command",
                {"cmd": install_cmd},
                ctx
            )
            
            install_log = install_result.output if install_result.ok else (
                install_result.output + f"\nError: {install_result.error}"
            )
            full_log.append(f"=== Install ===\n{install_log}")
            
            if not install_result.ok:
                final_log = "\n\n".join(full_log)
                outputs["execution_log.txt"] = final_log
                state.save_artifact("execution_log.txt", final_log)
                return StageResult(
                    success=False,
                    outputs=outputs,
                    error="Install failed",
                    message=f"Installation failed:\n{install_log[:300]}"
                )
            
            print("     ✓ Install successful")
        else:
            print("  ⏭️  Step 2: No install command configured, skipping")
        
        # Step 3: Run tests (required)
        run_test_cmd_template = commands.get("run_test", "") or ""
        if not run_test_cmd_template.strip():
            run_test_cmd_template = "PYTHONPATH={project_root}:$PYTHONPATH pytest -q {test_file_path}"
        
        run_test_cmd = run_test_cmd_template.format(
            op=state.op,
            soc=state.soc,
            vendor=state.vendor,
            arch=state.arch,
            project_root=str(state.project_root),
            test_file_path=test_file_path,
            target=state.target,
            target_slug=state.target_slug,
        )
        
        print(f"  🏃 Step 3: Running tests...")
        print(f"     Command: {run_test_cmd}")
        
        run_result = self.tool_runner.execute(
            "exec_command",
            {"cmd": run_test_cmd},
            ctx
        )
        
        run_log = run_result.output if run_result.ok else (
            run_result.output + f"\nError: {run_result.error}"
        )
        full_log.append(f"=== Run Tests ===\n{run_log}")
        
        # Save full execution log
        final_log = "\n\n".join(full_log)
        outputs["execution_log.txt"] = final_log
        state.save_artifact("execution_log.txt", final_log)
        
        if run_result.ok:
            print("     ✓ Tests executed successfully")
            exit_code = "0"
            message = "测试执行成功"
        else:
            print("     ⚠️ Tests executed with errors")
            exit_code = "1"
            message = "测试未完全通过（退出码 1），详见 execution_log.txt"
        
        outputs["exit_code.txt"] = exit_code
        state.save_artifact("exit_code.txt", exit_code)
        
        return StageResult(
            success=True,
            outputs=outputs,
            message=message
        )
    
    def _resolve_test_file_path(self, state: WorkflowState, default_path: str) -> str:
        def _has_match(root: Path, pattern: str) -> bool:
            if not pattern:
                return False
            if any(ch in pattern for ch in "*?[]"):
                return any(root.glob(pattern))
            return (root / pattern).exists()

        plan_path = state.artifacts_dir / "design_test_plan" / "current_test_plan.json"
        if not plan_path.exists():
            return default_path
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            return default_path

        if not isinstance(plan, dict):
            return default_path

        test_files = plan.get("test_files") or {}
        project_root = Path(state.project_root)
        if isinstance(test_files, dict):
            single = test_files.get("single")
            if isinstance(single, str) and single.strip():
                if _has_match(project_root, single):
                    return single

        groups = plan.get("groups") or []
        group_files = {}
        if isinstance(test_files, dict):
            group_files = test_files.get("groups") or {}

        if isinstance(groups, list):
            for group in groups:
                if not isinstance(group, dict):
                    continue
                group_id = group.get("group_id")
                test_file = group.get("test_file")
                if group_id and test_file:
                    group_files.setdefault(group_id, test_file)

        active_order = plan.get("active_group_order") or []
        if isinstance(active_order, list) and active_order:
            index = max(0, getattr(state, "epoch_current", 1) - 1)
            if index < len(active_order):
                group_id = active_order[index]
                if isinstance(group_id, str) and group_id in group_files:
                    candidate = group_files[group_id]
                    if _has_match(project_root, candidate):
                        return candidate
            all_pattern = test_files.get("all_pattern") if isinstance(test_files, dict) else None
            if isinstance(all_pattern, str) and all_pattern.strip():
                if _has_match(project_root, all_pattern):
                    return all_pattern

        all_pattern = test_files.get("all_pattern") if isinstance(test_files, dict) else None
        if isinstance(all_pattern, str) and all_pattern.strip():
            if _has_match(project_root, all_pattern):
                return all_pattern

        default_override = test_files.get("default") if isinstance(test_files, dict) else None
        if isinstance(default_override, str) and default_override.strip():
            if _has_match(project_root, default_override):
                return default_override

        return default_path

    def get_config(self) -> StageConfig:
        return self.config
