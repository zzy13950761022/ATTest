"""
Base class for workflow stages (sub-agents).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from ..llm import LLMClient, ChatResponse
from ..tools import ToolRunner, ToolContext
from .state import WorkflowState
from ..config import load_config
from ..utils import slugify_target, ensure_parent
from ..session import append_message


@dataclass
class StageConfig:
    """Configuration for a workflow stage."""
    name: str
    display_name: str
    description: str
    prompt_template: str
    input_artifacts: List[str]  # Required artifacts from previous stages
    output_artifacts: List[str]  # Artifacts this stage produces
    tools: List[str]  # Tool names available to this stage
    allow_skip: bool = False


@dataclass
class StageResult:
    """Result of stage execution."""
    success: bool
    outputs: Dict[str, Any]  # Generated artifacts
    error: Optional[str] = None
    message: str = ""


class Stage(ABC):
    """
    Base class for all workflow stages.
    Each stage is a sub-agent with specific prompt and tools.
    """
    
    # Subclasses should define this
    config: StageConfig
    
    def __init__(self, llm: LLMClient, tool_runner: ToolRunner):
        self.llm = llm
        self.tool_runner = tool_runner
    
    def validate_inputs(self, state: WorkflowState) -> bool:
        """Check if all required input artifacts are available."""
        for artifact_name in self.config.input_artifacts:
            if state.load_artifact(artifact_name) is None:
                return False
        return True
    
    def render_prompt(self, state: WorkflowState) -> str:
        """
        Render prompt template with artifacts and context.
        
        Subclasses can override for custom rendering.
        """
        # Load configuration
        from ..config import load_config
        config = load_config()
        target = getattr(state, "target", state.op)
        target_slug = getattr(state, "target_slug", None) or slugify_target(target)
        
        # Load input artifacts
        artifact_values = {}
        for artifact_name in self.config.input_artifacts:
            artifact_values[artifact_name.replace('.', '_')] = \
                state.load_artifact(artifact_name) or ""
        
        # Get user feedback for this stage
        feedback_for_stage = [f.feedback for f in state.user_feedback 
                             if f.stage == self.config.name]
        user_feedback_str = "\n".join(feedback_for_stage) if feedback_for_stage else "No feedback"
        
        # Build project context
        project_config = config.get("project", {})
        # 提供默认模板，防止配置缺失导致路径为空
        test_file_template = project_config.get("test_file_template") or "tests/test_{target_slug}.py"
        test_file_path = test_file_template.format(
            op=state.op,
            target=target,
            target_slug=target_slug,
        )
        output_binary_template = project_config.get("output_binary_template", "")
        output_binary = output_binary_template.format(
            op=state.op,
            target=target,
            target_slug=target_slug,
        )

        extra_vars = self.get_prompt_vars(
            state=state,
            target=target,
            target_slug=target_slug,
            test_file_path=test_file_path,
            output_binary=output_binary,
        )
        if extra_vars is None:
            extra_vars = {}
        
        # Render template with enhanced context
        return self.config.prompt_template.format(
            op=state.op,
            arch=state.arch,
            soc=state.soc,
            vendor=state.vendor,
            target=target,
            target_fqn=target,
            target_slug=target_slug,
            project_root=str(state.project_root),
            test_file_path=test_file_path,
            output_binary=output_binary,
            user_feedback=user_feedback_str,
            **artifact_values,
            **extra_vars
        )

    def get_prompt_vars(
        self,
        state: WorkflowState,
        target: str,
        target_slug: str,
        test_file_path: str,
        output_binary: str,
    ) -> Dict[str, Any]:
        """
        Provide extra variables for prompt formatting.
        Override in subclasses when needed.
        """
        return {}
    
    def execute(self, state: WorkflowState) -> StageResult:
        """
        Execute this stage.
        
        Default implementation:
        1. Validate inputs
        2. Render prompt
        3. Call LLM with tools
        4. Save outputs
        5. Return result
        """
        # Validate
        if not self.validate_inputs(state):
            missing = [a for a in self.config.input_artifacts 
                      if state.load_artifact(a) is None]
            return StageResult(
                success=False,
                outputs={},
                error=f"Missing required artifacts: {missing}"
            )
        
        # Render prompt
        try:
            prompt = self.render_prompt(state)
        except Exception as e:
            return StageResult(
                success=False,
                outputs={},
                error=f"Prompt rendering failed: {e}"
            )
        

        # 确保测试文件的父目录存在，避免工具调用时因目录缺失报错
        try:
            cfg = load_config()
            project_config = cfg.get("project", {})
            test_file_template = project_config.get("test_file_template") or "tests/test_{target_slug}.py"
            target = getattr(state, "target", state.op)
            target_slug = getattr(state, "target_slug", None) or slugify_target(target)
            test_file_path = test_file_template.format(
                op=state.op,
                target=target,
                target_slug=target_slug,
            )
            ensure_parent(Path(state.project_root) / test_file_path)
        except Exception:
            # 静默忽略，避免影响主流程
            pass

        # Get tool schemas if needed
        tool_schemas = None
        if self.config.tools:
            available_tools = {t for t in self.tool_runner.registry.all().keys()}
            requested_tools = set(self.config.tools)
            if not requested_tools.issubset(available_tools):
                unknown = requested_tools - available_tools
                return StageResult(
                    success=False,
                    outputs={},
                    error=f"Unknown tools requested: {unknown}"
                )
            # Filter tool schemas
            all_schemas = self.tool_runner.registry.to_llm_schema()
            tool_schemas = [s for s in all_schemas 
                           if s["function"]["name"] in self.config.tools]
        
        # Multi-turn tool calling loop (like chat mode)
        session_id = getattr(state, "workflow_id", "workflow")
        # 首条提示记录到日志
        messages = [{"role": "user", "content": prompt}]
        append_message(
            session_id=session_id,
            role="user",
            content={"stage": self.config.name, "prompt": prompt},
            workspace=str(state.workspace),
            stage=self.config.name,
        )
        outputs = {}
        # 将工具执行目录固定到 project_root，确保写文件时自动创建缺失父目录
        ctx = ToolContext(cwd=str(state.project_root), auto_approve=True)
        # 提前保存本阶段推荐的写入路径，方便缺参时给出提示
        fallback_test_path = locals().get("test_file_path", "")
        block_edit_counts = {}
        
        max_iterations = 100  # Same as chat mode
        for iteration in range(max_iterations):
            # Call LLM
            try:
                response = self.llm.chat(messages, tools=tool_schemas)
            except Exception as e:
                return StageResult(
                    success=False,
                    outputs={},
                    error=f"LLM call failed: {e}"
                )
            
            # Add assistant response to messages
            assistant_msg = {
                "role": "assistant",
                "content": response.content,
                # deepseek-reasoner 等模型需要带 reasoning_content 字段
                "reasoning_content": getattr(response, "reasoning_content", "") or ""
            }
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)
            append_message(
                session_id=session_id,
                role="assistant",
                content=assistant_msg,
                workspace=str(state.workspace),
                stage=self.config.name,
            )
            
            # If no tool calls, we're done
            if not response.has_tool_calls():
                # Save final content if any
                if response.content and self.config.output_artifacts:
                    for artifact in self.config.output_artifacts:
                        if artifact not in outputs:
                            outputs[artifact] = response.content
                            state.save_artifact(artifact, response.content)
                break
            
            # Handle tool calls (tool_display shows all details)
            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                import json
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}
                
                # 如果 write_file 调用缺少必需参数，直接返回带提示的错误，避免无意义的反复重试
                if tool_name == "write_file":
                    missing = []
                    if not tool_args.get("path"):
                        missing.append("path")
                    if "content" not in tool_args or tool_args.get("content") is None:
                        missing.append("content")
                    if missing:
                        hint = (
                            "write_file 需要同时提供 path 和 content。"
                            + (f" 推荐路径: {fallback_test_path}" if fallback_test_path else "")
                        )
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"Error: missing {missing}. {hint}".strip()
                        }
                        messages.append(tool_msg)
                        append_message(
                            session_id=session_id,
                            role="tool",
                            content=tool_msg,
                            workspace=str(state.workspace),
                            stage=self.config.name,
                        )
                        continue
                    # generate_code 阶段禁止覆盖已存在文件，要求用局部修改
                    if self.config.name == "generate_code":
                        file_arg = tool_args.get("path")
                        if file_arg:
                            target_path = Path(ctx.cwd) / file_arg
                            if target_path.exists() and target_path.is_file():
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": (
                                        "Error: target file already exists, do not overwrite. "
                                        "Use replace_block (or read_file/part_read + replace_in_file) for incremental edits."
                                    )
                                }
                                messages.append(tool_msg)
                                append_message(
                                    session_id=session_id,
                                    role="tool",
                                    content=tool_msg,
                                    workspace=str(state.workspace),
                                    stage=self.config.name,
                                )
                                continue

                if self.config.name == "generate_code" and tool_name == "replace_block":
                    block_id = (tool_args.get("block_id") or "").strip()
                    if block_id:
                        count = block_edit_counts.get(block_id, 0)
                        if count >= 1:
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": (
                                    f"Error: block {block_id} already edited in this stage. "
                                    "Limit to one replace per block; defer further changes to analyze_results."
                                )
                            }
                            messages.append(tool_msg)
                            append_message(
                                session_id=session_id,
                                role="tool",
                                content=tool_msg,
                                workspace=str(state.workspace),
                                stage=self.config.name,
                            )
                            continue
                        block_edit_counts[block_id] = count + 1
                
                # Execute tool (tool_display handles all display)
                result = self.tool_runner.execute(tool_name, tool_args, ctx)
                
                # Track outputs - save artifact immediately for write_file
                if tool_name == "write_file" and "path" in tool_args and result.ok:
                    filename = tool_args["path"]
                    content = tool_args.get("content", "")
                    outputs[filename] = content
                    # Save to state immediately
                    state.save_artifact(filename, content)
                
                # Add tool result to messages for next iteration
                # Include error message so LLM knows what went wrong
                content_for_llm = result.output if result.ok else f"Error: {result.error}"
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": content_for_llm
                }
                messages.append(tool_msg)
                append_message(
                    session_id=session_id,
                    role="tool",
                    content=tool_msg,
                    workspace=str(state.workspace),
                    stage=self.config.name,
                )
        
        if iteration >= max_iterations - 1:
            print(f"\n  ⚠️  Max iterations ({max_iterations}) reached")
        
        return StageResult(
            success=True,
            outputs=outputs,
            message=response.content if response.content else "Stage completed"
        )
    
    @abstractmethod
    def get_config(self) -> StageConfig:
        """Return stage configuration. Subclasses must implement."""
        pass
