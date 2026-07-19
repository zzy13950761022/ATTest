# AGENTS.md

## Project overview

Dual-package CLI tool for LLM-guided test generation for Python APIs (PyTorch/TensorFlow) and Ascend C operators.

- **Package**: `src/attest_cli/` — primary, full-featured (Python + Ascend workflows)
- **Package**: `src/testagent_cli/` — legacy, Python-only, mostly a subset of `attest_cli`
- **CLI entry points**: both `attest` and `testagent` (registered in `pyproject.toml`)
- **Config file**: `~/.attest_cli/config.json` (legacy: `~/.testagent_cli/config.json`)
- **Python requirement**: `>=3.11`

## Install & setup

```bash
pip install -e .
```

The config file is auto-created on first run; it embeds an API key and model URL. Override config dir via env vars: `ATTEST_CONFIG_DIR` or `TESTAGENT_CONFIG_DIR`.

## Commands

```bash
attest chat --workspace <dir>           # interactive chat mode (tool use)
attest run -f <func> --workspace <dir>  # Python 7-stage workflow
attest run -f <func> --mode full-auto --epoch 3  # multi-epoch full-auto
attest ascend-ut run --op-path <path> --project-root <dir>  # Ascend C UT workflow
attest config set KEY VALUE             # nested dot-key (e.g. api.model)
attest config list                      # dump full config
```

Workflow interactive commands: `/next`, `/regenerate`, `/goto <stage>`, `/retry`, `/quit`, `/help`, `/status`.

## Testing

No pytest config or test runner is configured. Tests are standalone scripts run with `python` directly:

```bash
python test_milestone1.py    # Core framework (state, stage, engine)
python test_smoke.py         # Quick smoke with mock LLM
python test_custom_commands.py
python test_workflow_e2e.py
python test_chat.py
python test_llm.py
python test_ascend_workflow.py
```

Tests import from `src/` by doing `sys.path.insert(0, 'src')` — no editable-install dependency.

## Architecture notes

- **Workflow engine** (`attest_cli/workflow/engine.py`): orchestrates the 7-stage pipeline with state persistence (`.testagent/state.json`), artifact versioning (`.testagent/artifacts/<stage>/vN_*`), and epoch-based feedback loops in full-auto mode
- **Workflow profiles** (`profiles.py`): `python` uses standard stages; `ascend_ut` uses ascend-specific stages — profile is selected by `workflow_kind` parameter
- **Epoch system**: full-auto mode iterates codegen→execute→analyze→back to codegen. Auto-stops on stable failure signatures or repeated block-level errors across consecutive epochs
- **Block-level editing**: test code uses markers like `HEADER`, `CASE_*`, `FOOTER` — each block can be edited at most once per epoch
- **Tool system**: file ops (`read_file`, `write_file`, `replace_in_file`, `search`), command execution (`exec_command`), and Python introspection (`inspect_python`) exposed to LLM via function-calling schemas
- The `analyze_results` stage produces an `analysis_plan.json` that drives the next `generate_code` epoch — failures, deferred cases, stop recommendations

## Config quirks

- Config uses dot-notation nested keys: `set api.model deepseek-v4-pro`
- Default config includes a hardcoded DeepSeek API key in `src/attest_cli/config.py:32` — system reads from disk, not code, on subsequent runs
- `commands.run_test` template supports variables: `{project_root}`, `{test_file_path}`, `{target_slug}`
- Ascend profile commands use `{op_name}`, `{soc}` template variables