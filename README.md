# ATTest

A controllable CLI agent for generating and running tests for Python operators and APIs (such as PyTorch and TensorFlow) through a fixed, human-approved workflow.

## ✨ Key Features

### Dual-Mode Operation
- **🔧 Tool Mode (Chat)**: Ad hoc queries and file operations
- **🔄 Workflow Mode**: A 7-stage intelligent test generation pipeline

### Workflow Highlights
- **7-stage pipeline**: Function understanding → requirements generation → test plan design → code generation → test execution → result analysis → report generation
- **Smart feedback**: Supports natural-language feedback and special commands (`/regenerate`, `/goto`, `/retry`)
- **Persistent state**: Supports interruption recovery, with versioned artifact management
- **Custom execution**: Configurable pytest commands, environment variables, and dependency installation

---

## 🚀 Quick Start

### Installation

```bash
cd ATTest
pip install -e .
```

### Basic Usage

#### Chat Mode (Ad Hoc Tasks)

```bash
attest chat --workspace ~/my-project
```

#### Workflow Mode (Full Test Generation)

```bash
# Interactive mode (recommended)
attest run -f torch.nn.functional.relu --workspace ./my-project/torch.nn.functional.relu

# Fully automatic mode
attest run -f torch.add --mode full-auto --workspace ./my-project/torch.add

# Fully automatic mode with multi-round iteration (3 rounds)
attest run -f torch.add --mode full-auto --epoch 3

# Specify the project root when it differs from workspace
attest run -f torch.add --workspace ./my-project --project-root ./my-project/src

# Resume an interrupted workflow
attest run -f torch.add --resume
```

---

## 📖 Documentation

- **[WORKFLOW_GUIDE.md](./WORKFLOW_GUIDE.md)** - Complete usage guide (recommended)
  - Detailed workflow mode walkthrough
  - Custom build command configuration
  - Testing and debugging
  - Extension and customization

- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - Quick reference card
  - Common command cheat sheet
  - Workflow interactive commands
  - Configuration examples

- **[USAGE.md](./USAGE.md)** - Chat mode usage guide

---

## 🎯 Workflow Stages

```
╔═══════════════════════════════════════════════════╗
║   ATTest Workflow - torch.add (python)           ║
╠═══════════════════════════════════════════════════╣
║ [●] 1. Understand Function                       ║
║ [●] 2. Generate Requirements                     ║
║ [▶] 3. Design Test Plan        ← Current Stage   ║
║ [ ] 4. Generate Code                             ║
║ [ ] 5. Execute Tests                             ║
║ [ ] 6. Analyze Results                           ║
║ [ ] 7. Generate Report                           ║
╚═══════════════════════════════════════════════════╝
```

After each stage completes, you can:
- Press `Enter` to continue
- Use `/regenerate` to regenerate the current stage
- Use `/goto <stage>` to jump to another stage
- Provide natural-language feedback for smart interpretation

---

## 🔧 Custom Build Commands

When you need to adjust the pytest command or environment variables, edit `~/.attest_cli/config.json`:

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

**Available variables**:
- `{target}` / `{target_slug}` - Target function FQN and its slug
- `{project_root}` - Project root directory
- `{test_file_path}` - Generated pytest file path

See [WORKFLOW_GUIDE.md - Custom Build Commands](./WORKFLOW_GUIDE.md#custom-build-commands) for details.

---

## 📋 Command Reference

### Configuration Management

```bash
attest config list              # Show all configuration values
attest config set KEY VALUE     # Set a configuration value
attest config get KEY           # Get a configuration value
```

### Session Management

```bash
attest sessions list            # List all sessions
attest sessions clear ID        # Clear a specific session
```

### Chat Mode

```bash
attest chat [--workspace DIR] [--auto-approve]
```

### Workflow Mode

```bash
attest run -f package.module:function \
  [--workspace DIR] \
  [--project-root DIR] \
  [--mode interactive|full-auto] \
  [--epoch N] \
  [--resume]
```

**Arguments**:
- `-f, --func`: Fully qualified target function name (required)
- `--workspace`: Working directory, defaults to the current directory
- `--project-root`: Project root directory, defaults to the same value as `workspace`
- `--mode`: Run mode, either `interactive` or `full-auto`, defaults to `interactive`
- `--epoch`: Number of iteration rounds in full-auto mode; after `analyze_results`, the workflow loops back to `generate_code`; defaults to `1`
- `--resume`: Resume an interrupted workflow

---

## 🛠️ Extension and Customization

### Add a New Tool

1. Create a tool class in `src/attest_cli/tools/builtin.py`
2. Register it in `src/attest_cli/tools/runner.py`
3. Add it to a stage's `tools` list

### Modify a Stage Prompt

Edit the corresponding stage file, for example `src/attest_cli/workflow/stages/requirements.py`, and update `_get_prompt_template()`.

### Add a New Stage

1. Create `src/attest_cli/workflow/stages/your_stage.py`
2. Register it in `stages/__init__.py`
3. Add it to `STAGE_NAMES` in `workflow/engine.py`

See [WORKFLOW_GUIDE.md - Extension and Customization](./WORKFLOW_GUIDE.md#extension-and-customization) for details.

---

## 📂 Project Structure

```
ATTest/
├── src/attest_cli/
│   ├── cli.py                    # CLI entry point
│   ├── config.py                 # Configuration management
│   ├── llm.py                    # LLM client
│   ├── session.py                # Session management
│   ├── chat.py                   # Chat mode
│   ├── tools/                    # Tool system
│   │   ├── base.py
│   │   ├── builtin.py
│   │   └── runner.py
│   └── workflow/                 # Workflow engine
│       ├── engine.py             # Main controller
│       ├── state.py              # State management
│       ├── stage.py              # Stage base class
│       ├── supervisor.py         # Feedback interpretation
│       ├── display.py            # Progress display
│       └── stages/               # Individual stages
│           ├── understand.py
│           ├── requirements.py
│           ├── planning.py
│           ├── codegen.py
│           ├── execution.py
│           ├── analysis.py
│           └── report.py
├── WORKFLOW_GUIDE.md             # Complete usage guide
├── QUICK_REFERENCE.md            # Quick reference
└── USAGE.md                      # Chat mode guide
```

---

## 🧪 Tests

```bash
# Test the core framework
python test_milestone1.py

# Test custom commands
python test_custom_commands.py
```

---

## 📝 Examples

### Fully Automatic Workflow for a PyTorch Operator

```bash
attest run -f torch.add --workspace ~/my-project --mode full-auto
```

Automatically generates pytest cases, executes them, and produces a test report.

### Workflow with a Custom Pytest Command

```bash
attest config set commands.run_test "PYTHONPATH={project_root}:$PYTHONPATH pytest -q {test_file_path} -k gpu"
attest run -f torch.add --workspace ~/my-project --mode full-auto
```

Uses a custom command to run the generated tests, which can be reused in a virtual environment or CI pipeline.

### Quick Checks in Chat Mode

```bash
attest chat --workspace ~/my-project
```

Temporarily inspect or analyze files, or generate small testing snippets.

---

## 🤝 Contributing

Contributions are welcome. Main extension points include:
- New tools (file operations, analysis utilities, and so on)
- New workflow stages (performance testing, coverage analysis, and so on)
- Improved prompt templates
- Better error handling

---

## 📄 License

MIT

---

## 🙏 Acknowledgements

Inspired by:
- **CliAgent** - Flow-first design
- **Cougar-CLI** - Config and session management
- **mini-kode** - Permission-aware tool execution

---

**Quick start**: See [WORKFLOW_GUIDE.md](./WORKFLOW_GUIDE.md) for detailed usage.
