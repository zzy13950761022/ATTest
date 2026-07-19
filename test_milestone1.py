"""
Test script for Milestone 1: Core Framework
Tests WorkflowState, Stage base class, and basic workflow execution.
"""
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, 'src')

from testagent_cli.workflow import WorkflowState, Stage, StageConfig, StageResult, WorkflowEngine
from testagent_cli.llm import LLMClient
from testagent_cli.tools import build_default_registry, ToolRunner, ToolContext

print("Testing Milestone 1: Core Framework")
print("=" * 60)

# Create temporary workspace
workspace = tempfile.mkdtemp(prefix="testagent_m1_test_")
print(f"Test workspace: {workspace}\n")

try:
    # Test 1: WorkflowState creation and persistence
    print("[Test 1] WorkflowState creation and persistence")
    state = WorkflowState(workspace, op="conv2d", arch="x86")
    print(f"✓ Created state: {state.workflow_id}")
    
    # Save artifact
    state.save_artifact("requirements.md", "Test requirements content")
    print(f"✓ Saved artifact: requirements.md")
    
    # Persist state
    state.persist()
    print(f"✓ Persisted state to disk")
    
    # Load state back
    loaded_state = WorkflowState.load(workspace)
    assert loaded_state is not None
    assert loaded_state.workflow_id == state.workflow_id
    print(f"✓ Loaded state from disk")
    
    # Load artifact
    content = loaded_state.load_artifact("requirements.md")
    assert content == "Test requirements content"
    print(f"✓ Loaded artifact from disk\n")
    
    # Test 2: Stage base class
    print("[Test 2] Stage base class")
    
    class MockStage(Stage):
        def __init__(self, llm, tool_runner):
            self.config = StageConfig(
                name="mock_stage",
                display_name="Mock Stage",
                description="A mock stage for testing",
                prompt_template="Test prompt for {op} on {arch}",
                input_artifacts=[],
                output_artifacts=["output.txt"],
                tools=[]
            )
            super().__init__(llm, tool_runner)
        
        def get_config(self):
            return self.config
    
    # Create mock LLM and tools
    llm = LLMClient(
        base_url="https://api.deepseek.com/v1",
        api_key="mock-key",
        model="deepseek-chat"
    )
    registry = build_default_registry()
    runner = ToolRunner(registry)
    
    mock_stage = MockStage(llm, runner)
    print(f"✓ Created mock stage: {mock_stage.config.name}")
    
    # Test prompt rendering
    prompt = mock_stage.render_prompt(state)
    assert "conv2d" in prompt
    assert "x86" in prompt
    print(f"✓ Rendered prompt successfully\n")
    
    # Test 3: Progress display
    print("[Test 3] Progress display")
    from testagent_cli.workflow.display import show_progress
    
    state.current_stage = "generate_requirements"
    state.stage_index = 1
    state.record_stage_completion("understand_function", "completed")
    
    show_progress(state, WorkflowEngine.STAGE_NAMES)
    print("✓ Progress display rendered\n")
    
    # Test 4: State recovery
    print("[Test 4] State recovery")
    state.current_stage = "design_test_plan"
    state.stage_index = 2
    state.save_artifact("test_plan.md", "Test plan content")
    state.add_feedback("Add more edge cases", "regenerate")
    state.persist()
    
    # Simulate crash and recovery
    recovered_state = WorkflowState.load(workspace)
    assert recovered_state is not None
    assert recovered_state.current_stage == "design_test_plan"
    assert recovered_state.stage_index == 2
    assert len(recovered_state.user_feedback) == 1
    print(f"✓ State recovered successfully")
    print(f"  Current stage: {recovered_state.current_stage}")
    print(f"  Feedback count: {len(recovered_state.user_feedback)}\n")
    
    print("=" * 60)
    print("✅ All Milestone 1 tests passed!")
    print(f"\nWorkflow structure created at:")
    print(f"  {workspace}/.testagent/")
    print(f"    ├── state.json")
    print(f"    ├── artifacts/")
    print(f"    └── logs/")
    
finally:
    # Cleanup
    print(f"\nCleaning up test workspace...")
    if Path(workspace).exists():
        shutil.rmtree(workspace)
        print(f"✓ Cleaned up: {workspace}")
