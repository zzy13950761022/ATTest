"""
Workflow engine - main orchestrator for stage-based test generation.
"""
from typing import Any, Dict, List, Optional
from pathlib import Path
import hashlib
import json
import sys
import traceback

from ..llm import LLMClient
from ..tools import build_default_registry, ToolRunner
from .state import WorkflowState
from .stage import Stage, StageResult
from .profiles import get_workflow_profile
from .display import (
    show_progress,
    show_stage_output,
    collect_feedback,
    show_error,
    show_help
)


class WorkflowEngine:
    """
    Main workflow orchestrator.
    Manages stage execution, user feedback, and state persistence.
    """
    
    # Define workflow stages (will be populated with actual stages later)
    STAGE_NAMES = [
        "understand_function",
        "generate_requirements",
        "design_test_plan",
        "generate_code",
        "execute_tests",
        "analyze_results",
        "generate_report"
    ]
    
    def __init__(
        self,
        llm: LLMClient,
        workspace: str,
        op: str,
        arch: str,
        soc: str = "ascend910b",
        vendor: str = "custom",
        project_root: Optional[str] = None,
        target: Optional[str] = None,
        resume: bool = False,
        epochs: int = 1,
        workflow_kind: str = "python",
        op_path: str = "",
        repo_name: str = "",
        category: str = "",
        op_name: str = "",
        generation_mode: str = "ut_generate",
        coverage_mode: str = "single_run",
        enabled_layers: Optional[List[str]] = None,
        skill_root: str = "",
        workers: int = 1,
    ):
        self.llm = llm
        self.workspace = Path(workspace)
        self.op = op
        self.arch = arch
        self.soc = soc
        self.vendor = vendor
        self.target = target or op
        self.project_root = Path(project_root) if project_root else self.workspace
        self.profile = get_workflow_profile(workflow_kind)
        self.STAGE_NAMES = list(self.profile.stage_names)

        # Initialize state
        if resume:
            self.state = WorkflowState.load(str(self.workspace))
            if self.state is None:
                raise ValueError("No existing workflow found to resume")
        else:
            self.state = WorkflowState.load_or_create(
                str(self.workspace),
                op,
                arch,
                soc,
                vendor,
                str(self.project_root),
                self.target,
                epoch_total=epochs,
                epoch_current=1,
                workflow_kind=workflow_kind,
                op_path=op_path,
                repo_name=repo_name,
                category=category,
                op_name=op_name,
                generation_mode=generation_mode,
                coverage_mode=coverage_mode,
                enabled_layers=enabled_layers,
                skill_root=skill_root,
            )

        # Keep target fields in sync for legacy state files
        self.state.target = getattr(self.state, "target", self.target)
        self.state.target_slug = getattr(self.state, "target_slug", self.state.op)
        self.state.workflow_kind = getattr(self.state, "workflow_kind", workflow_kind)
        self.state.op_path = getattr(self.state, "op_path", op_path)
        self.state.repo_name = getattr(self.state, "repo_name", repo_name)
        self.state.category = getattr(self.state, "category", category)
        self.state.op_name = getattr(self.state, "op_name", op_name)
        self.state.generation_mode = getattr(self.state, "generation_mode", generation_mode)
        self.state.coverage_mode = getattr(self.state, "coverage_mode", coverage_mode)
        self.state.enabled_layers = getattr(self.state, "enabled_layers", enabled_layers or [])
        self.state.skill_root = getattr(self.state, "skill_root", skill_root)
        # Epoch config (respect persisted values on resume; CLI epochs only used when creating)
        if resume:
            if getattr(self.state, "epoch_total", None) is None:
                self.state.epoch_total = max(1, epochs)
            if getattr(self.state, "epoch_current", None) is None:
                self.state.epoch_current = 1
        else:
            # load_or_create already set for fresh state; ensure defaults exist for old state files
            self.state.epoch_total = max(1, getattr(self.state, "epoch_total", epochs))
            self.state.epoch_current = max(1, getattr(self.state, "epoch_current", 1))
        # Parallel workers config (respect persisted value on resume; mirror epoch pattern)
        self.state.workers = max(1, getattr(self.state, "workers", workers))

        # Initialize tool system
        self.tool_registry = build_default_registry()
        self.tool_runner = ToolRunner(self.tool_registry)
        
        # Initialize stages (will be populated in subclass or dynamically)
        self.stages: Dict[str, Stage] = {}
        self._register_stages()
    
    def _register_stages(self):
        """
        Register all workflow stages.
        """
        self.stages = self.profile.build_stages(self.llm, self.tool_runner)
        # Inject parallel worker count into the continuous generation loop stage if supported
        loop_stage = self.stages.get("generate_code")
        if loop_stage is not None and hasattr(loop_stage, "set_workers"):
            loop_stage.set_workers(getattr(self.state, "workers", 1))

    def _load_analysis_plan(self) -> Optional[Dict[str, Any]]:
        plan_path = self.state.artifacts_dir / "analyze_results" / "current_analysis_plan.json"
        if not plan_path.exists():
            return None
        try:
            return json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _compute_failure_signatures(self, plan: Dict[str, Any]) -> Dict[str, str]:
        failures = plan.get("failures") or []
        if not isinstance(failures, list):
            failures = []

        failure_keys: List[str] = []
        error_types = set()
        for item in failures:
            if not isinstance(item, dict):
                continue
            test_name = str(item.get("test") or "")
            block_id = str(item.get("block_id") or "")
            error_type = str(item.get("error_type") or "")
            failure_keys.append(f"{test_name}|{block_id}|{error_type}")
            if error_type:
                error_types.add(error_type)

        failure_keys.sort()
        error_type_list = sorted(error_types)

        failure_sig = ""
        error_sig = ""
        if failure_keys:
            failure_sig = hashlib.sha1("|".join(failure_keys).encode("utf-8")).hexdigest()
        if error_type_list:
            error_sig = hashlib.sha1("|".join(error_type_list).encode("utf-8")).hexdigest()

        return {"failure_signature": failure_sig, "error_signature": error_sig}

    def _extract_block_errors(self, plan: Dict[str, Any]) -> Dict[str, List[str]]:
        failures = plan.get("failures") or []
        if not isinstance(failures, list):
            failures = []

        block_errors: Dict[str, set] = {}
        for item in failures:
            if not isinstance(item, dict):
                continue
            block_id = str(item.get("block_id") or "").strip()
            error_type = str(item.get("error_type") or "").strip()
            if not block_id or not error_type:
                continue
            block_errors.setdefault(block_id, set()).add(error_type)

        return {block_id: sorted(list(errors)) for block_id, errors in block_errors.items()}

    def _save_analysis_plan(self, plan: Dict[str, Any]) -> None:
        plan_path = self.state.artifacts_dir / "analyze_results" / "current_analysis_plan.json"
        try:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps(plan, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            return

    def _apply_skip_blocks(self, plan: Dict[str, Any], skip_blocks: List[str], reason: str) -> bool:
        if not skip_blocks:
            return False
        failures = plan.get("failures") or []
        if not isinstance(failures, list):
            return False
        deferred = plan.get("deferred") or []
        if not isinstance(deferred, list):
            deferred = []

        skip_set = set(skip_blocks)
        remaining = []
        skipped = []
        for item in failures:
            if not isinstance(item, dict):
                continue
            block_id = str(item.get("block_id") or "").strip()
            if block_id and block_id in skip_set:
                test_name = str(item.get("test") or "")
                error_type = str(item.get("error_type") or "")
                if test_name or block_id:
                    deferred.append(
                        {
                            "test": test_name,
                            "reason": f"{block_id}: {reason}".strip(),
                        }
                    )
                skipped.append(
                    {
                        "test": test_name,
                        "block_id": block_id,
                        "error_type": error_type,
                        "reason": reason,
                    }
                )
                continue
            remaining.append(item)

        if not skipped:
            return False

        plan["failures"] = remaining
        plan["deferred"] = deferred
        plan["stop_recommended"] = False
        plan["stop_reason"] = ""
        plan["skipped_blocks"] = skipped
        self._save_analysis_plan(plan)
        return True

    def _check_auto_stop(self) -> Optional[str]:
        plan = self._load_analysis_plan()
        if not plan:
            return None
        if not isinstance(plan, dict):
            return None

        plan_stop_recommended = bool(plan.get("stop_recommended"))
        plan_stop_reason = str(plan.get("stop_reason") or "").strip()
        failures = plan.get("failures") or []
        if not failures:
            self.state.last_failure_signature = ""
            self.state.last_error_signature = ""
            self.state.last_block_errors = {}
            if plan_stop_recommended:
                return plan_stop_reason or "analysis_plan stop recommended"
            return None

        # If all failures are pure coverage infrastructure errors, do not auto-stop
        # so the LLM can attempt to fix the coverage configuration in the next epoch.
        if all(
            isinstance(item, dict) and item.get("error_type") == "CoverageCollectionError"
            for item in failures
        ):
            self.state.last_failure_signature = ""
            self.state.last_error_signature = ""
            self.state.last_block_errors = {}
            if plan_stop_recommended:
                return plan_stop_reason or "analysis_plan stop recommended"
            return None

        signatures = self._compute_failure_signatures(plan)
        failure_sig = signatures["failure_signature"]
        error_sig = signatures["error_signature"]

        prev_failure_sig = self.state.last_failure_signature
        prev_error_sig = self.state.last_error_signature
        prev_block_errors = getattr(self.state, "last_block_errors", {}) or {}
        if not isinstance(prev_block_errors, dict):
            prev_block_errors = {}
        current_block_errors = self._extract_block_errors(plan)
        self.state.last_failure_signature = failure_sig
        self.state.last_error_signature = error_sig
        self.state.last_block_errors = current_block_errors

        if prev_failure_sig and failure_sig and failure_sig == prev_failure_sig:
            return "The set of failing cases did not change across two consecutive rounds; terminate automatically"

        if prev_error_sig and error_sig and error_sig == prev_error_sig:
            skip_blocks = []
            for block_id, errors in current_block_errors.items():
                prev_errors = set(prev_block_errors.get(block_id, []))
                if prev_errors.intersection(errors):
                    skip_blocks.append(block_id)
            if skip_blocks:
                skip_blocks = sorted(set(skip_blocks))
                self._apply_skip_blocks(
                    plan,
                    skip_blocks,
                    "The error type repeated across two consecutive rounds; skip this code block",
                )
                previously_skipped = plan.get("skipped_blocks") or []
                total_skipped = len(previously_skipped)
                total_cases = len(plan.get("cases") or [])
                if total_cases > 0 and total_skipped > max(total_cases // 2, 2):
                    return (
                        f"Block-skip budget exhausted: {total_skipped}/{total_cases} blocks skipped; "
                        "too many blocks have unresolvable errors; terminate automatically"
                    )
            return None

        if plan_stop_recommended:
            return plan_stop_reason or "analysis_plan stop recommended"
        return None
    
    def _restore_best_artifacts(self) -> None:
        best = int(getattr(self.state, "best_coverage_epoch", 0) or 0)
        current = int(getattr(self.state, "epoch_current", 1) or 1)
        if best == 0 or best == current:
            return
        artifacts_dir = self.state.artifacts_dir
        if not artifacts_dir.is_dir():
            return
        restored = 0
        for stage_dir in artifacts_dir.iterdir():
            if not stage_dir.is_dir():
                continue
            prefix = f"v{best}_"
            for versioned in stage_dir.iterdir():
                if not versioned.is_file() or not versioned.name.startswith(prefix):
                    continue
                name = versioned.name[len(prefix):]
                current_link = stage_dir / f"current_{name}"
                try:
                    if current_link.exists() or current_link.is_symlink():
                        current_link.unlink()
                    current_link.symlink_to(versioned.name)
                    restored += 1
                except OSError:
                    pass
        if restored:
            print(
                f"\n🔄 Epoch rollback: current_epoch={current} → best_epoch={best}; "
                f"re-linked {restored} artifact(s) in `{artifacts_dir.name}/`."
            )
        snap_dir = self.state.snapshots_dir / f"v{best}"
        if snap_dir.is_dir():
            manifest_path = artifacts_dir / "generate_code" / f"v{best}_generation_manifest.json"
            if manifest_path.exists():
                import json, shutil
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    project_root = Path(manifest.get("project_root", ""))
                    files = manifest.get("files") or []
                    file_count = 0
                    for entry in files:
                        rel = Path(entry.get("path", ""))
                        src = snap_dir / rel
                        dst = project_root / rel
                        if src.is_file():
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(src), str(dst))
                            file_count += 1
                    if file_count:
                        print(
                            f"\n🔄 Epoch rollback: restored {file_count} test file(s) "
                            f"from v{best} snapshot → `{project_root.name}/`."
                        )
                except (json.JSONDecodeError, OSError):
                    pass
    
    def get_stage(self, stage_name: str) -> Optional[Stage]:
        """Get stage by name."""
        return self.stages.get(stage_name)
    
    def run(self, mode: str = "interactive"):
        """
        Run the workflow.
        
        Args:
            mode: "interactive" or "full_auto"
        """
        self.state.mode = mode
        # Disable early stopping and clear historical early-stop reasons
        self.state.auto_stop_reason = ""
        # Safety: avoid repeated iterations beyond the configured round limit (resume uses the persisted round count)
        if mode in {"full-auto", "full_auto"}:
            self.state.epoch_total = max(1, getattr(self.state, "epoch_total", 1))
            self.state.epoch_current = max(1, getattr(self.state, "epoch_current", 1))
            if self.state.epoch_current > self.state.epoch_total:
                self.state.epoch_current = self.state.epoch_total
        
        # Show workflow start with panel
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.markup import escape as _re
            console = Console()
            console.print()
            console.print(Panel(
                f"[bold]Target:[/bold] {_re(str(self.state.target))}\n"
                f"[bold]Slug:[/bold] {_re(str(self.state.target_slug))}\n"
                f"[bold]Architecture:[/bold] {self.arch}\n"
                f"[bold]Mode:[/bold] {mode}\n"
                f"[bold]Workspace:[/bold] {_re(str(self.workspace))}",
                title="[bold cyan]Starting Test Generation Workflow[/bold cyan]",
                border_style="cyan",
                padding=(0, 2)
            ))
            console.print()
            use_rich = True
        except ImportError:
            print(f"\n=== Starting ATTest Workflow ===")
            print(f"Target: {self.state.target} (slug={self.state.target_slug})")
            print(f"Architecture: {self.arch}")
            print(f"Mode: {mode}")
            print(f"Workspace: {self.workspace}\n")
            use_rich = False
        
        # Main execution loop
        had_failure = False
        while not self.state.is_complete():
            # Show progress
            show_progress(self.state, self.STAGE_NAMES)
            
            # Get current stage
            stage = self.get_stage(self.state.current_stage)
            if stage is None:
                show_error(f"Stage '{self.state.current_stage}' not implemented yet")
                break
            
            # Show stage execution with panel
            if use_rich:
                console.print()
                console.print(Panel(
                    f"[bold]{stage.config.display_name}[/bold]",
                    border_style="yellow",
                    padding=(0, 2)
                ))
            else:
                print(f"\n--- Executing: {stage.config.display_name} ---")
            
            try:
                result = stage.execute(self.state)
            except Exception as exc:
                error_msg = f"Unhandled exception in stage '{self.state.current_stage}': {exc}"
                result = StageResult(success=False, outputs={}, error=error_msg)
                print(f"\n  ❌ {error_msg}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            
            # Handle result
            if not result.success:
                had_failure = True
                show_error(result.error or "Stage failed")
                self.state.record_stage_completion(
                    self.state.current_stage,
                    "failed",
                    result.error
                )
                
                if mode == "interactive":
                    action = self._handle_failure()
                    if action == "quit":
                        break
                    elif action == "retry":
                        continue
                    elif action == "skip":
                        self.state.advance_stage(self.STAGE_NAMES)
                else:
                    print(f"\n  ❌ Stage '{self.state.current_stage}' failed; workflow stopped.", file=sys.stderr)
                    break
            else:
                # Success
                self.state.record_stage_completion(
                    self.state.current_stage,
                    "completed"
                )
                show_stage_output(
                    stage.config.display_name,
                    result.message,
                    list(result.outputs.keys())
                )

                if result.complete_workflow:
                    self.state.current_stage = "complete"
                    self.state.stage_index = len(self.STAGE_NAMES)
                    self.state.persist()
                    break
                
                # Show an artifact summary
                if result.outputs:
                    print(f"\n  📦 Stage Outputs:")
                    for output_name in result.outputs.keys():
                        # Try to locate the saved file
                        artifact_path = self.state.artifacts_dir / self.state.current_stage / f"current_{output_name}"
                        if artifact_path.exists():
                            size = artifact_path.stat().st_size
                            print(f"    • {output_name} ({size} bytes)")
                        else:
                            print(f"    • {output_name}")
                
                # Interactive mode - get feedback
                if mode == "interactive":
                    action = self._collect_and_process_feedback()
                    if action == "quit":
                        break
                    elif action == "retry":
                        continue  # Retry same stage
                    elif action == "next":
                        self.state.advance_stage(self.STAGE_NAMES)
                    # For "goto", state.current_stage already updated
                else:
                    # Full-auto mode supports epoch-style iteration.
                    # Epoch boundary: either `analyze_results` (standard pipeline) or any
                    # stage with `is_epoch_boundary = True` (e.g. AscendGenerationAgentLoopStage).
                    _is_epoch_boundary = (
                        stage.config.name == "analyze_results"
                        or getattr(stage, "is_epoch_boundary", False)
                    )
                    if mode in {"full-auto", "full_auto"} and _is_epoch_boundary:
                        auto_stop_reason = self._check_auto_stop()
                        if auto_stop_reason:
                            self.state.auto_stop_reason = auto_stop_reason
                            self._restore_best_artifacts()
                            self.state.advance_stage(self.STAGE_NAMES)
                        elif getattr(self.state, "epoch_current", 1) < getattr(self.state, "epoch_total", 1):
                            self.state.epoch_current += 1
                            # Phase 2: if the reviewer requested planner augmentation
                            # for plateaued layers, re-enter design_test_plan instead of
                            # jumping straight back to generation.
                            _plan = self._load_analysis_plan() or {}
                            _replan = (
                                bool(_plan.get("replan_recommended"))
                                and "design_test_plan" in self.STAGE_NAMES
                            )
                            if _replan:
                                _layers = (_plan.get("augment_request") or {}).get("layers", [])
                                print(
                                    f"\n🔁 Iteration {self.state.epoch_current}/{self.state.epoch_total}: "
                                    f"augmenting test plan for layers {_layers} (reviewer requested re-plan)."
                                )
                                self.state.jump_to_stage("design_test_plan", self.STAGE_NAMES)
                            else:
                                print(
                                    f"\n🔁 Iteration {self.state.epoch_current}/{self.state.epoch_total}: return to the Generate Code stage to continue improving the tests."
                                )
                                self.state.jump_to_stage("generate_code", self.STAGE_NAMES)
                        else:
                            self._restore_best_artifacts()
                            self.state.advance_stage(self.STAGE_NAMES)
                    else:
                        self.state.advance_stage(self.STAGE_NAMES)
            
            # Persist state after each stage
            self.state.persist()
        
        # Workflow complete or stopped
        if self.state.is_complete():
            if self.state.auto_stop_reason:
                print(f"\n✅ Workflow completed (early-stop iterations): {self.state.auto_stop_reason}")
            else:
                print("\n✅ Workflow completed successfully!")
        elif had_failure:
            failed_stages = [
                r.stage for r in self.state.stage_history if r.status == "failed"
            ]
            print(f"\n❌ Workflow stopped due to failures in: {', '.join(failed_stages)}", file=sys.stderr)
            print("Check state.json and logs for details.", file=sys.stderr)
            self._restore_best_artifacts()
            self.state.persist()
            sys.exit(1)
        else:
            print("\n⏸️  Workflow paused. Resume with --resume flag.")
        
        # Final state save
        self.state.persist()
    
    def _collect_and_process_feedback(self) -> str:
        """
        Collect user feedback and decide action using SupervisorAgent.
        Returns: "next", "retry", "goto", or "quit"
        """
        feedback = collect_feedback()
        
        # Import and use supervisor
        from .supervisor import SupervisorAgent
        if not hasattr(self, 'supervisor'):
            self.supervisor = SupervisorAgent(self.llm)
        
        # Interpret feedback
        action = self.supervisor.interpret_feedback(
            feedback, 
            self.state,
            self.STAGE_NAMES
        )
        
        # Apply action
        if action.type == "continue":
            return "next"
        
        elif action.type == "retry":
            if action.context:
                print(f"  📝 Feedback recorded: {action.context}")
                self.state.add_feedback(action.context, "regenerate")
            return "retry"
        
        elif action.type == "goto":
            if action.target_stage:
                try:
                    self.state.jump_to_stage(action.target_stage, self.STAGE_NAMES)
                    print(f"  ↪ Jumping to stage: {action.target_stage}")
                    return "goto"
                except ValueError as e:
                    print(f"  Error: {e}")
                    return "retry"
            return "retry"
        
        elif action.type == "quit":
            return "quit"
        
        else:
            # Default to next
            return "next"
    
    def _handle_command(self, command: str) -> str:
        """
        Handle special commands.
        Returns: action to take
        """
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/next":
            return "next"
        
        elif cmd == "/regenerate" or cmd == "/retry":
            if len(parts) > 1:
                feedback_msg = " ".join(parts[1:])
                self.state.add_feedback(feedback_msg, "regenerate")
            return "retry"
        
        elif cmd == "/goto":
            if len(parts) < 2:
                print("  Usage: /goto <stage_name>")
                return self._collect_and_process_feedback()
            
            target_stage = parts[1]
            try:
                self.state.jump_to_stage(target_stage, self.STAGE_NAMES)
                return "goto"
            except ValueError as e:
                print(f"  Error: {e}")
                return self._collect_and_process_feedback()
        
        elif cmd == "/status":
            self._show_status()
            return self._collect_and_process_feedback()
        
        elif cmd == "/quit":
            return "quit"
        
        elif cmd == "/help":
            show_help()
            return self._collect_and_process_feedback()
        
        else:
            print(f"  Unknown command: {cmd}")
            print("  Type /help for available commands")
            return self._collect_and_process_feedback()
    
    def _handle_failure(self) -> str:
        """Handle stage failure in interactive mode."""
        print("\nStage failed. What would you like to do?")
        print("  [r] Retry")
        print("  [s] Skip and continue")
        print("  [q] Quit")
        
        choice = input("> ").strip().lower()
        if choice == "r":
            return "retry"
        elif choice == "s":
            return "skip"
        else:
            return "quit"
    
    def _show_status(self):
        """Show current workflow status."""
        print("\n📊 Workflow Status:")
        print(f"  ID: {self.state.workflow_id}")
        print(f"  Current Stage: {self.state.current_stage}")
        print(f"  Progress: {self.state.stage_index + 1}/{len(self.STAGE_NAMES)}")
        print(f"  Completed Stages: {len([r for r in self.state.stage_history if r.status == 'completed'])}")
        print(f"  Failed Stages: {len([r for r in self.state.stage_history if r.status == 'failed'])}")
        print(f"  User Feedback: {len(self.state.user_feedback)} items")
