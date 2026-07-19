"""
Workflow state management with persistence and artifact versioning.
"""
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field

from ..utils import slugify_target, ensure_parent


WORKFLOW_DIR_NAME = ".attest"
LEGACY_WORKFLOW_DIR_NAME = ".testagent"


@dataclass
class StageRecord:
    """Record of a completed stage."""
    stage: str
    status: str  # 'completed', 'failed', 'skipped'
    timestamp: str
    attempts: int = 1
    error: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class FeedbackRecord:
    """User feedback on a stage."""
    stage: str
    feedback: str
    timestamp: str
    action: str  # 'regenerate', 'retry', 'skip', 'continue'
    
    def to_dict(self):
        return asdict(self)


class WorkflowState:
    """
    Manages workflow state, artifacts, and history.
    Supports persistence and recovery.
    """
    
    def __init__(
        self, 
        workspace: str, 
        op: str, 
        arch: str,
        soc: str = "ascend910b",
        vendor: str = "custom",
        project_root: Optional[str] = None,
        target: Optional[str] = None,
        epoch_total: int = 1,
        epoch_current: int = 1,
        workflow_kind: str = "python",
        op_path: str = "",
        repo_name: str = "",
        category: str = "",
        op_name: str = "",
        generation_mode: str = "ut_generate",
        coverage_mode: str = "single_run",
        enabled_layers: Optional[List[str]] = None,
        skill_root: str = "",
    ):
        self.workspace = Path(workspace)
        self.op = op
        self.arch = arch
        self.soc = soc
        self.vendor = vendor
        self.project_root = Path(project_root) if project_root else self.workspace
        self.target = target or op
        self.target_slug = slugify_target(self.target)
        self.workflow_kind = workflow_kind
        self.op_path = op_path
        self.repo_name = repo_name
        self.category = category
        self.op_name = op_name
        self.generation_mode = generation_mode
        self.coverage_mode = coverage_mode
        self.enabled_layers = enabled_layers or []
        self.skill_root = skill_root
        self.workflow_id = str(uuid.uuid4())[:8]
        self.created_at = datetime.now().isoformat()
        
        # Workflow progress
        self.current_stage = "understand_function"
        self.stage_index = 0
        self.mode = "interactive"  # or "full_auto"
        self.epoch_total = max(1, epoch_total)
        self.epoch_current = max(1, epoch_current)
        self.last_failure_signature = ""
        self.last_error_signature = ""
        self.last_block_errors: Dict[str, List[str]] = {}
        self.auto_stop_reason = ""
        self.best_coverage_metrics: Dict[str, float] = {}
        self.coverage_no_improvement_rounds = 0
        
        # Data storage
        self.artifacts: Dict[str, Any] = {}
        self.stage_history: List[StageRecord] = []
        self.user_feedback: List[FeedbackRecord] = []
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories."""
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        (self.workflow_dir / "artifacts").mkdir(exist_ok=True)
        (self.workflow_dir / "logs").mkdir(exist_ok=True)

    @property
    def workflow_dir(self) -> Path:
        """Path to workflow directory."""
        return self.workspace / WORKFLOW_DIR_NAME
    
    @property
    def state_file(self) -> Path:
        """Path to state JSON file."""
        return self.workflow_dir / "state.json"
    
    @property
    def artifacts_dir(self) -> Path:
        """Path to artifacts directory."""
        return self.workflow_dir / "artifacts"
    
    def save_artifact(self, name: str, content: Any, version: bool = True):
        """
        Save artifact to both memory and disk.
        
        Args:
            name: Artifact name (e.g., "requirements.md")
            content: Artifact content
            version: If True, create versioned copy
        """
        # Save to memory
        self.artifacts[name] = content
        
        # Determine storage location
        if name.endswith(('.md', '.txt', '.log', '.c', '.sh', '.py', '.json')):
            # Save as file
            stage_dir = self.artifacts_dir / self.current_stage
            stage_dir.mkdir(exist_ok=True)
            
            if version:
                # Find next version number
                version_num = 1
                while (stage_dir / f"v{version_num}_{name}").exists():
                    version_num += 1
                
                artifact_path = stage_dir / f"v{version_num}_{name}"
                # Also create/update "current" symlink
                current_path = stage_dir / f"current_{name}"
            else:
                artifact_path = stage_dir / name
                current_path = None
            
            # Write file
            ensure_parent(artifact_path)
            artifact_path.write_text(str(content), encoding='utf-8')
            
            # Show the saved path
            relative_path = artifact_path.relative_to(self.workspace)
            print(f"  📄 Saved: {relative_path}")
            
            # Update symlink
            if current_path:
                ensure_parent(current_path)
                if current_path.exists() or current_path.is_symlink():
                    current_path.unlink()
                current_path.symlink_to(artifact_path.name)
    
    def load_artifact(self, name: str, version: str = "current") -> Optional[Any]:
        """
        Load artifact from memory or disk.
        
        Args:
            name: Artifact name
            version: "current" or specific version like "v1"
        
        Returns:
            Artifact content or None
        """
        # Try memory first
        if name in self.artifacts:
            return self.artifacts[name]
        
        # Try disk
        stage_dirs = [self.artifacts_dir / self.current_stage]
        stage_dirs.extend(path for path in self.artifacts_dir.iterdir() if path.is_dir())

        for stage_dir in stage_dirs:
            if version == "current":
                artifact_path = stage_dir / f"current_{name}"
                if artifact_path.is_symlink():
                    artifact_path = stage_dir / artifact_path.readlink()
            else:
                artifact_path = stage_dir / f"{version}_{name}"

            if artifact_path.exists():
                content = artifact_path.read_text(encoding='utf-8')
                self.artifacts[name] = content
                return content
        
        return None
    
    def record_stage_completion(self, stage: str, status: str, error: Optional[str] = None):
        """Record stage execution result."""
        # Check if we're retrying
        attempts = 1
        for record in reversed(self.stage_history):
            if record.stage == stage:
                attempts = record.attempts + 1
                break
        
        self.stage_history.append(StageRecord(
            stage=stage,
            status=status,
            timestamp=datetime.now().isoformat(),
            attempts=attempts,
            error=error
        ))
    
    def add_feedback(self, content: str, action: str):
        """Record user feedback."""
        self.user_feedback.append(FeedbackRecord(
            stage=self.current_stage,
            feedback=content,
            action=action,
            timestamp=datetime.now().isoformat()
        ))
    
    def jump_to_stage(self, stage_name: str, stages: List[str]):
        """Jump to a specific stage."""
        if stage_name in stages:
            self.current_stage = stage_name
            self.stage_index = stages.index(stage_name)
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    def advance_stage(self, stages: List[str]):
        """Move to next stage."""
        if self.stage_index < len(stages) - 1:
            self.stage_index += 1
            self.current_stage = stages[self.stage_index]
        else:
            self.current_stage = "complete"
    
    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.current_stage == "complete"
    
    def persist(self):
        """Save current state to disk."""
        state_file = self.state_file
        
        # Ensure parent directory exists
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state_data = {
            "workflow_id": self.workflow_id,
            "created_at": self.created_at,
            "op": self.op,
            "arch": self.arch,
            "soc": self.soc,
            "vendor": self.vendor,
            "project_root": str(self.project_root),
            "target": self.target,
            "target_slug": self.target_slug,
            "workflow_kind": self.workflow_kind,
            "op_path": self.op_path,
            "repo_name": self.repo_name,
            "category": self.category,
            "op_name": self.op_name,
            "generation_mode": self.generation_mode,
            "coverage_mode": self.coverage_mode,
            "enabled_layers": self.enabled_layers,
            "skill_root": self.skill_root,
            "current_stage": self.current_stage,
            "stage_index": self.stage_index,
            "mode": self.mode,
            "epoch_total": self.epoch_total,
            "epoch_current": self.epoch_current,
            "last_failure_signature": self.last_failure_signature,
            "last_error_signature": self.last_error_signature,
            "last_block_errors": self.last_block_errors,
            "auto_stop_reason": self.auto_stop_reason,
            "best_coverage_metrics": self.best_coverage_metrics,
            "coverage_no_improvement_rounds": self.coverage_no_improvement_rounds,
            "artifacts": self.artifacts,
            "stage_history": [r.to_dict() for r in self.stage_history],
            "user_feedback": [f.to_dict() for f in self.user_feedback],
        }
        
        state_file.write_text(
            json.dumps(state_data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
    
    @classmethod
    def load(cls, workspace: str) -> Optional['WorkflowState']:
        """Load existing state from disk."""
        workspace_path = Path(workspace)
        workflow_dir = workspace_path / WORKFLOW_DIR_NAME
        legacy_dir = workspace_path / LEGACY_WORKFLOW_DIR_NAME

        # Auto-migrate legacy workspace directory name when needed.
        if not workflow_dir.exists() and legacy_dir.exists():
            legacy_dir.rename(workflow_dir)

        state_file = workflow_dir / "state.json"
        if not state_file.exists():
            return None
        
        data = json.loads(state_file.read_text(encoding='utf-8'))
        
        # Create instance with new parameters
        state = cls(
            workspace, 
            data["op"], 
            data["arch"],
            soc=data.get("soc", "ascend910b"),
            vendor=data.get("vendor", "custom"),
            project_root=data.get("project_root"),
            target=data.get("target"),
            epoch_total=data.get("epoch_total", 1),
            epoch_current=data.get("epoch_current", 1),
            workflow_kind=data.get("workflow_kind", "python"),
            op_path=data.get("op_path", ""),
            repo_name=data.get("repo_name", ""),
            category=data.get("category", ""),
            op_name=data.get("op_name", ""),
            generation_mode=data.get("generation_mode", "ut_generate"),
            coverage_mode=data.get("coverage_mode", "single_run"),
            enabled_layers=data.get("enabled_layers", []),
            skill_root=data.get("skill_root", ""),
        )
        state.workflow_id = data["workflow_id"]
        state.created_at = data["created_at"]
        state.current_stage = data["current_stage"]
        state.stage_index = data["stage_index"]
        state.mode = data["mode"]
        state.target_slug = data.get("target_slug", slugify_target(state.target))
        state.workflow_kind = data.get("workflow_kind", "python")
        state.op_path = data.get("op_path", "")
        state.repo_name = data.get("repo_name", "")
        state.category = data.get("category", "")
        state.op_name = data.get("op_name", "")
        state.generation_mode = data.get("generation_mode", "ut_generate")
        state.coverage_mode = data.get("coverage_mode", "single_run")
        state.enabled_layers = data.get("enabled_layers", []) or []
        state.skill_root = data.get("skill_root", "")
        state.artifacts = data.get("artifacts", {})
        state.stage_history = [StageRecord(**r) for r in data.get("stage_history", [])]
        state.user_feedback = [FeedbackRecord(**f) for f in data.get("user_feedback", [])]
        state.last_failure_signature = data.get("last_failure_signature", "")
        state.last_error_signature = data.get("last_error_signature", "")
        state.last_block_errors = data.get("last_block_errors", {}) or {}
        state.auto_stop_reason = data.get("auto_stop_reason", "")
        state.best_coverage_metrics = data.get("best_coverage_metrics", {}) or {}
        state.coverage_no_improvement_rounds = int(data.get("coverage_no_improvement_rounds", 0) or 0)
        
        return state
    
    @classmethod
    def load_or_create(
        cls, 
        workspace: str, 
        op: str, 
        arch: str,
        soc: str = "ascend910b",
        vendor: str = "custom",
        project_root: Optional[str] = None,
        target: Optional[str] = None,
        epoch_total: int = 1,
        epoch_current: int = 1,
        workflow_kind: str = "python",
        op_path: str = "",
        repo_name: str = "",
        category: str = "",
        op_name: str = "",
        generation_mode: str = "ut_generate",
        coverage_mode: str = "single_run",
        enabled_layers: Optional[List[str]] = None,
        skill_root: str = "",
    ) -> 'WorkflowState':
        """Load existing state or create new one."""
        state = cls.load(workspace)
        if state is None:
            state = cls(
                workspace,
                op,
                arch,
                soc,
                vendor,
                project_root,
                target,
                epoch_total=epoch_total,
                epoch_current=epoch_current,
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
        return state
