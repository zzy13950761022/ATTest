# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATTest is a CLI agent for LLM-guided test generation targeting Python operators (PyTorch, TensorFlow) and Ascend C operators. It runs a 7-stage workflow pipeline driven by an OpenAI-compatible LLM (default: DeepSeek).

Two CLI entry points registered in `pyproject.toml`:
- `attest` — primary, full-featured
- `testagent` — legacy alias (Python workflow only)

## Setup & Commands

```bash
pip install -e .
```

Tests are standalone scripts (no pytest runner config); run them directly:

```bash
python test_milestone1.py       # Core framework: state, stage, engine
python test_smoke.py            # Quick smoke test with mock LLM
python test_custom_commands.py
python test_workflow_e2e.py
python test_chat.py
python test_llm.py
python test_ascend_workflow.py
```

Batch testing:

```bash
python batch_test_torch.py                             # 5 epochs, 51 PyTorch modules
bash scripts/run_pynguinml_tensorflow_66_121.sh        # TensorFlow batch
```

No linting or build pipeline is configured.

## Architecture

### Packages

`src/attest_cli/` is the primary package. `src/testagent_cli/` is a legacy alias that mirrors a subset of it.

### Core Flow

```
CLI (cli.py)
  └── WorkflowEngine (workflow/engine.py)
        ├── loads a Profile (workflow/profiles.py): "python" or "ascend_ut"
        ├── iterates over Stages
        │     each Stage:
        │       - renders a prompt using context from prior stage artifacts
        │       - calls LLMClient (llm.py) with function-calling schemas
        │       - executes tool calls via ToolRunner (tools/runner.py)
        │       - saves output as versioned artifacts in .attest/artifacts/<stage>/
        └── persists progress to .attest/state.json
```

**Python workflow stages** (`workflow/stages/`): understand → requirements → planning → codegen → execution → analysis → report

**Ascend C workflow stages** (`workflow/ascend_stages.py`, ~3600 lines): understand → analysis_plan → generate_cases → generate_code → compile_and_run → extract_coverage → report

### Full-Auto Epoch Loop

In full-auto mode, after `analysis`, the engine loops back to `codegen` for up to N epochs. It auto-stops on stable failure signatures or repeated block-level errors. Test code uses block markers (`HEADER`, `CASE_*`, `FOOTER`) — each block is editable at most once per epoch (`block_utils.py`).

### Tool System

Tools available to the LLM via function-calling: `read_file`, `write_file`, `replace_in_file`, `search` (ripgrep), `exec_command` (shell), `inspect_python` (signature/docs/source). Defined in `tools/builtin.py`, executed by `tools/runner.py`.

## Configuration

Config file: `~/.attest_cli/config.json` (auto-created on first run). Override location with `ATTEST_CONFIG_DIR`.

```bash
attest config set api.model deepseek-chat
attest config get api.model
attest config list
```

Key config paths:
- `api.model` / `api.base_url` / `api.api_key` — LLM connection
- `commands.run_test` — pytest command template; supports `{project_root}`, `{test_file_path}`, `{target_slug}`
- `profiles.ascend_ut.skill_root` — path to `ascendc-ut-develop/`
- `profiles.ascend_ut.commands.*` — Ascend build.sh invocation templates

Debug logging: set `ATTEST_DEBUG_LLM=1`.

## State & Artifacts

- `.attest/state.json` — workflow progress (workspace-local)
- `.attest/artifacts/<stage>/vN_*` — versioned stage outputs
- `~/.attest_cli/sessions/` — chat history
- Generated tests default to `tests/test_<target_slug>.py` (configurable)

## Key Files

| File | Purpose |
|------|---------|
| `src/attest_cli/cli.py` | CLI commands (typer): `run`, `chat`, `ascend-ut run`, `config` |
| `src/attest_cli/workflow/engine.py` | Orchestrates stage execution, epoch loop, state persistence |
| `src/attest_cli/workflow/profiles.py` | Defines the `python` and `ascend_ut` stage sequences |
| `src/attest_cli/workflow/ascend_stages.py` | All 7 Ascend C stages (~3600 lines) |
| `src/attest_cli/workflow/stages/` | Python workflow stages (one file per stage) |
| `src/attest_cli/llm.py` | LLMClient wrapping OpenAI SDK |
| `src/attest_cli/tools/` | Tool base, built-in tools, runner |
| `src/attest_cli/block_utils.py` | Block-marker parsing for epoch-level test editing |
