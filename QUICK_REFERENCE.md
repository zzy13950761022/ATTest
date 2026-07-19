# ATTest Quick Reference

## 🚀 Common Commands

```bash
# Default interactive mode (chat)
attest [--workspace DIR] [--auto-approve]

# Explicit chat mode
attest chat [--workspace DIR] [--auto-approve]

# Workflow mode (interactive) - short flags
attest run -f package.module:function [--workspace DIR] [--project-root DIR]

# Workflow mode (interactive) - long flags
attest run --func package.module:function [--workspace DIR]

# Workflow mode (fully automatic)
attest run -f package.module:function --mode full-auto

# Fully automatic mode with multi-round iteration (for example, 3 rounds)
attest run -f package.module:function --mode full-auto --epoch 3

# Resume an interrupted workflow
attest run -f package.module:function --resume

# Configuration management
attest config list
attest config set KEY VALUE
attest config get KEY

# Session management
attest sessions list
attest sessions clear <session_id>
```

## 📝 Workflow Interactive Commands

After each stage completes, you can use:

| Command | Description |
|------|------|
| `Enter` | Approve and continue to the next stage |
| `/next` | Same as above |
| `/regenerate` | Regenerate the current stage |
| `/retry` | Regenerate with optional feedback, such as `/retry cover empty tensors` |
| `/goto <stage>` | Jump to a target stage, such as `/goto generate_code` |
| `/status` | Show workflow status |
| `/help` | Show help information |
| `/quit` | Exit the workflow |
| `<natural language>` | Provide feedback interpreted automatically by `SupervisorAgent` |

## 🔧 Custom Build Commands

Edit `~/.attest_cli/config.json`:

```json
{
  "api": {
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your-api-key",
    "temperature": 0.2,
    "max_tokens": 4096
  },
  "preferences": {
    "auto_approve": false
  },
  "project": {
    "root": ".",
    "test_file_template": "tests/test_{target_slug}.py",
    "build_dir": "",
    "output_binary_template": ""
  },
  "commands": {
    "compile": "",
    "install": "",
    "run_test": "PYTHONPATH={project_root}:$PYTHONPATH pytest -q {test_file_path}"
  }
}
```

Available variables:
- `{target}` / `{target_slug}` - Target function FQN and its slug
- `{project_root}` - Project root directory
- `{test_file_path}` - Generated pytest file path

## 🎯 Workflow Stages

```
1. understand_function    → Analyze the Python target
2. generate_requirements  → Generate requirements
3. design_test_plan       → Design the test plan
4. generate_code          → Generate pytest code
5. execute_tests          → Run pytest
6. analyze_results        → Analyze results
7. generate_report        → Generate the report
```

## 📂 Artifact Locations

```
workspace/
├── .attest/
│   ├── state.json                    # Workflow state
│   ├── artifacts/                    # Per-stage artifacts with versioning
│   │   ├── understand_function/
│   │   │   ├── current_function_doc.md    # Symlink to the current version
│   │   │   └── v1_function_doc.md         # Versioned storage
│   │   ├── generate_requirements/
│   │   │   ├── current_requirements.md
│   │   │   └── v1_requirements.md
│   │   └── ...
│   └── logs/                         # Log directory
├── tests/test_<target_slug>.py       # Generated pytest file
└── Optional project files
```

## 🛠️ Quick Customization

### Modify a Stage Prompt

```bash
vi src/attest_cli/workflow/stages/requirements.py
```

Edit `_get_prompt_template()`.

### Add a New Tool

1. Add the class in `src/attest_cli/tools/builtin.py`
2. Register it in `src/attest_cli/tools/runner.py`
3. Use it in a stage's `tools` list

### Debugging

```bash
# View workflow state
cat workspace/.attest/state.json

# View artifacts
ls workspace/.attest/artifacts/

# Unit tests
pytest test_workflow_e2e.py -q
pytest test_smoke.py -q
```

## ⚡ Examples

### Standard Usage

```bash
attest run -f torch.nn.functional.relu --workspace ~/my-proj
```

### Override the Custom Pytest Command

```bash
# 1. Configure
attest config set commands.run_test "PYTHONPATH={project_root}:$PYTHONPATH pytest -q {test_file_path} -k gpu"

# 2. Run
attest run -f torch.add --mode full-auto
```

### Modify Requirements Mid-Workflow

```
After stage 2 completes:
> The requirements are too simple. Add concurrency tests and performance tests.

The workflow continues and regenerates the requirements.
```

---

See [WORKFLOW_GUIDE.md](./WORKFLOW_GUIDE.md) for the full documentation.
