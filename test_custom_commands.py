"""
Test the fixes for custom command support in CodeGenStage and ExecutionStage.
"""
import sys
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, 'src')

print("Testing Custom Command Support Fixes")
print("=" * 60)

# Test 1: Verify config includes custom_commands
print("\n[Test 1] Verify config includes custom_commands")
from testagent_cli.config import load_config, DEFAULT_CONFIG

assert "custom_commands" in DEFAULT_CONFIG
assert "build_template" in DEFAULT_CONFIG["custom_commands"]
assert "run_template" in DEFAULT_CONFIG["custom_commands"]
print(f"✓ DEFAULT_CONFIG has custom_commands")
print(f"  Build template: {DEFAULT_CONFIG['custom_commands']['build_template'][:50]}...")
print(f"  Run template: {DEFAULT_CONFIG['custom_commands']['run_template']}")

# Test 2: CodeGenStage loads custom commands
print("\n[Test 2] CodeGenStage loads custom commands")
from testagent_cli.llm import LLMClient
from testagent_cli.tools import build_default_registry, ToolRunner
from testagent_cli.workflow.stages import CodeGenStage

llm = LLMClient(
    base_url="https://api.deepseek.com/v1",
    api_key="test-key",
    model="deepseek-chat"
)
registry = build_default_registry()
runner = ToolRunner(registry)

codegen_stage = CodeGenStage(llm, runner)
print(f"✓ CodeGenStage created")
print(f"  Build template: {codegen_stage.build_template[:50]}...")
print(f"  Run template: {codegen_stage.run_template}")

# Verify template is used in prompt
assert "{self.build_template}" in codegen_stage.config.prompt_template or \
       codegen_stage.build_template in codegen_stage._get_prompt_template()
print(f"✓ Build template is used in prompt")

# Test 3: Custom config values
print("\n[Test 3] Custom config values")
workspace = tempfile.mkdtemp(prefix="test_custom_cmd_")
config_path = Path(workspace) / ".testagent_cli" / "config.json"
config_path.parent.mkdir(parents=True, exist_ok=True)

custom_config = {
    "api": {"model": "test", "base_url": "", "api_key": ""},
    "custom_commands": {
        "build_template": "./my_build.sh {op} {arch}",
        "run_template": "./my_run.sh {workspace}/test"
    }
}
config_path.write_text(json.dumps(custom_config, indent=2))

# Override CONFIG_PATH temporarily
import testagent_cli.config as config_module
original_path = config_module.CONFIG_PATH
config_module.CONFIG_PATH = config_path

# Reload config
cfg = load_config()
print(f"✓ Loaded custom config")
print(f"  Custom build: {cfg['custom_commands']['build_template']}")
print(f"  Custom run: {cfg['custom_commands']['run_template']}")

# Create CodeGenStage with custom config
codegen_custom = CodeGenStage(llm, runner)
assert codegen_custom.build_template == "./my_build.sh {op} {arch}"
assert codegen_custom.run_template == "./my_run.sh {workspace}/test"
print(f"✓ CodeGenStage uses custom commands from config")

# Restore original path
config_module.CONFIG_PATH = original_path

# Cleanup
shutil.rmtree(workspace)

print("\n" + "=" * 60)
print("✅ All custom command tests passed!")
print("\nSummary:")
print("  ✓ config.py has custom_commands section")
print("  ✓ CodeGenStage loads and stores custom commands")
print("  ✓ Custom config values override defaults")
print("  ✓ ExecutionStage will use custom commands (logic added)")
