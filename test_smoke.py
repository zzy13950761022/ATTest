"""
Quick smoke test to verify workflow engine works with mock LLM.
Run with: python test_smoke.py
"""
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from testagent_cli.llm import ChatResponse
from testagent_cli.workflow import WorkflowEngine


class SimpleMockLLM:
    """Very simple mock for quick testing."""
    
    def __init__(self, model="mock", base_url="", api_key=""):
        self.model = model
    
    def chat(self, messages, tools=None):
        """Return canned responses."""
        content = messages[-1]["content"] if messages else ""
        
        # Just return simple content
        if "intent" in content.lower():
            response = '{"intent": "approve", "context": ""}'
        else:
            response = "# Mock Output\n\nThis is a mock response."
        
        return ChatResponse(
            content=response,
            role="assistant",
            tool_calls=[]
        )


def smoke_test():
    """Run a basic smoke test."""
    print("🧪 Running Phase 2 Smoke Test...\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"✓ Created test workspace: {tmpdir}")
        
        # Create engine
        llm = SimpleMockLLM()
        engine = WorkflowEngine(
            llm=llm,
            workspace=tmpdir,
            op="test_add",
            arch="x86",
            resume=False
        )
        print("✓ WorkflowEngine initialized")
        
        # Check state
        assert engine.state is not None
        assert engine.state.op == "test_add"
        assert engine.state.arch == "x86"
        print("✓ Workflow state correct")
        
        # Check stages
        assert len(engine.stages) == 7
        expected = [
            "understand_function",
            "generate_requirements",
            "design_test_plan",
            "generate_code",
            "execute_tests",
            "analyze_results",
            "generate_report"
        ]
        for stage_name in expected:
            assert stage_name in engine.stages
        print(f"✓ All 7 stages registered: {list(engine.stages.keys())}")
        
        # Check supervisor is available
        from testagent_cli.workflow.supervisor import SupervisorAgent
        supervisor = SupervisorAgent(llm)
        action = supervisor.parse_command("/next", engine.STAGE_NAMES)
        assert action.type == "continue"
        print("✓ Supervisor working")
        
        # Verify state persistence
        engine.state.persist()
        state_file = Path(tmpdir) / ".testagent" / "state.json"
        assert state_file.exists()
        print(f"✓ State persisted to {state_file}")
        
        print("\n✅ All smoke tests passed!")
        print("\nPhase 2 implementation is ready for testing with real LLM.")
        return True


if __name__ == "__main__":
    try:
        smoke_test()
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
