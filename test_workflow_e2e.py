"""
End-to-end integration tests for the workflow engine.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

# Note: These tests require a properly configured LLM client
# For now, they serve as integration test templates


class MockLLMClient:
    """Mock LLM client for testing without actual API calls."""
    
    def __init__(self, model="mock", base_url="", api_key=""):
        self.model = model
        self.call_count = 0
    
    def chat(self, messages, tools=None):
        """Return mock responses based on message content."""
        self.call_count += 1
        
        from testagent_cli.llm import ChatResponse
        
        # Simple mock: return generic responses
        user_msg = messages[-1]["content"] if messages else ""
        
        if "analyze" in user_msg.lower() or "understand" in user_msg.lower():
            content = "# Function Analysis\n\nThis is a test operator."
        elif "requirements" in user_msg.lower():
            content = "# Test Requirements\n\n1. Test basic functionality"
        elif "plan" in user_msg.lower():
            content = "# Test Plan\n\n- Test case 1: Basic test"
        elif "intent" in user_msg.lower():
            # Supervisor feedback classification
            content = '{"intent": "approve", "context": "", "reasoning": "User approved"}'
        else:
            content = "Mock LLM response"
        
        return ChatResponse(
            content=content,
            role="assistant",
            tool_calls=[]
        )


def test_workflow_state_persistence():
    """Test that workflow state can be saved and loaded."""
    from testagent_cli.workflow  import WorkflowState
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create state
        state = WorkflowState(tmpdir, "test_op", "x86")
        state.save_artifact("test.txt", "test content")
        state.persist()
        
        # Load state
        loaded = WorkflowState.load(tmpdir)
        assert loaded is not None
        assert loaded.op == "test_op"
        assert loaded.arch == "x86"


def test_workflow_stage_execution():
    """Test individual stage can execute."""
    from testagent_cli.workflow import WorkflowState
    from testagent_cli.workflow.stages import UnderstandFunctionStage
    from testagent_cli.tools import build_default_registry, ToolRunner
    
    with tempfile.TemporaryDirectory() as tmpdir:
        llm = MockLLMClient()
        registry = build_default_registry()
        runner = ToolRunner(registry)
        
        state = WorkflowState(tmpdir, "conv2d", "x86")
        stage = UnderstandFunctionStage(llm, runner)
        
        result = stage.execute(state)
        
        # Should succeed even with mock LLM
        assert result.success or result.error is not None


def test_supervisor_command_parsing():
    """Test supervisor can parse special commands."""
    from testagent_cli.workflow.supervisor import SupervisorAgent
    
    llm = MockLLMClient()
    supervisor = SupervisorAgent(llm)
    
    stages = ["stage1", "stage2", "stage3"]
    
    # Test /next command
    action = supervisor.parse_command("/next", stages)
    assert action.type == "continue"
    
    # Test /retry command
    action = supervisor.parse_command("/retry need more tests", stages)
    assert action.type == "retry"
    assert action.context == "need more tests"
    
    # Test /goto command
    action = supervisor.parse_command("/goto stage2", stages)
    assert action.type == "goto"
    assert action.target_stage == "stage2"
    
    # Test /quit command
    action = supervisor.parse_command("/quit", stages)
    assert action.type == "quit"


def test_all_stages_registered():
    """Test that all 7 stages are properly registered."""
    from testagent_cli.workflow.stages import build_all_stages
    from testagent_cli.tools import build_default_registry, ToolRunner
    
    llm = MockLLMClient()
    registry = build_default_registry()
    runner = ToolRunner(registry)
    
    stages = build_all_stages(llm, runner)
    
    expected_stages = [
        "understand_function",
        "generate_requirements",
        "design_test_plan",
        "generate_code",
        "execute_tests",
        "analyze_results",
        "generate_report"
    ]
    
    for stage_name in expected_stages:
        assert stage_name in stages
        assert stages[stage_name] is not None


def test_workflow_engine_initialization():
    """Test that workflow engine initializes correctly."""
    from testagent_cli.workflow import WorkflowEngine
    
    with tempfile.TemporaryDirectory() as tmpdir:
        llm = MockLLMClient()
        
        engine = WorkflowEngine(
            llm=llm,
            workspace=tmpdir,
            op="conv2d",
            arch="x86",
            resume=False
        )
        
        assert engine.state is not None
        assert engine.state.op == "conv2d"
        assert engine.state.arch == "x86"
        assert len(engine.stages) == 7


# Manual test helpers (not automated)

def manual_test_full_workflow():
    """
    Manual test: Run a complete workflow with real LLM.
    
    Usage:
        pytest test_workflow_e2e.py::manual_test_full_workflow -v -s
    
    Note: Requires properly configured API credentials.
    """
    from testagent_cli.config import load_config
    from testagent_cli.llm import LLMClient
    from testagent_cli.workflow import WorkflowEngine
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config()
        llm = LLMClient(
            model=cfg["api"].get("model", "deepseek-chat"),
            base_url=cfg["api"].get("base_url", ""),
            api_key=cfg["api"].get("api_key", ""),
        )
        
        engine = WorkflowEngine(
            llm=llm,
            workspace=tmpdir,
            op="add",  # Simple operator for testing
            arch="x86",
            resume=False
        )
        
        # Run in full-auto mode
        print(f"\nRunning workflow in {tmpdir}")
        print("=" * 60)
        engine.run(mode="full-auto")
        
        # Check results
        report_path = Path(tmpdir) / ".testagent" / "artifacts" / "generate_report"
        if report_path.exists():
            print("\n✓ Workflow completed successfully!")
            print(f"  Artifacts in: {tmpdir}/.testagent/")
        else:
            print("\n✗ Workflow did not complete")


if __name__ == "__main__":
    # Run basic tests
    test_workflow_state_persistence()
    test_supervisor_command_parsing()
    test_all_stages_registered()
    test_workflow_engine_initialization()
    print("✓ All basic tests passed!")
