# TestAgent: A Workflow-Oriented CLI for LLM-Guided Test Generation

## Abstract
TestAgent is a command-line tool that orchestrates large language models (LLMs) to generate, execute, and refine test suites for Python APIs, with a focus on numerical and ML libraries such as PyTorch and TensorFlow. The system provides a structured, multi-stage workflow that separates intent understanding, requirement specification, test planning, code generation, execution, analysis, and reporting. TestAgent couples this workflow with a tool-driven execution layer that exposes controlled file and process operations, artifact versioning, and state persistence for resumption and reproducibility. This paper describes the architecture, workflow design, and key innovations, including multi-epoch feedback loops, block-level incremental editing, plan-driven test scope management, and selective log ingestion to mitigate prompt-length constraints. The result is a practical, extensible tool demo that balances automation with control and traceability, suitable for reproducible evaluation in research and engineering settings.

## 1. Introduction
Automated test generation for complex Python APIs requires more than a single-pass code synthesis. The target APIs often involve intricate parameter constraints, side effects, and environment-dependent behavior (e.g., device availability or compilation toolchains). TestAgent addresses this by decomposing test generation into a sequence of explicit stages that produce structured artifacts and support iterative refinement. The system is designed as a CLI agent with a workflow engine, enabling both interactive and full-automation modes while maintaining reproducibility and auditability through persistent state and artifact versioning.

## 2. System Overview
TestAgent consists of three primary layers:

1. **Workflow Engine**: A stage-based orchestrator that coordinates the end-to-end pipeline, supports resumption, and manages artifacts and history.
2. **Tool Execution Layer**: A registry of file, search, patch, execution, and Python inspection tools exposed to the LLM via function-calling schemas.
3. **CLI and Configuration**: A Typer-based CLI that launches workflows, with configurable build/run commands and environment settings.

The workflow runs inside a project workspace and writes all stage artifacts under a `.testagent/` directory. Each artifact is versioned and a `current_*` pointer is maintained for the most recent output. A state file records the workflow ID, stage index, artifacts, and historical actions, enabling interruption and resume.

## 3. Workflow Design
The workflow is a fixed seven-stage pipeline:

1. **Understand Function**: Inspect the target API (signature, documentation, and source if needed) to establish semantic context.
2. **Generate Requirements**: Produce a concise requirements document listing constraints, edge cases, and environment dependencies.
3. **Design Test Plan**: Create a machine-readable plan (`test_plan.json`) and a brief summary (`test_plan.md`) that serve as the sole specification for code generation.
4. **Generate Code**: Produce pytest code using a structured block template and tool-mediated file edits.
5. **Execute Tests**: Run configurable compile/install/test commands and capture logs and exit codes.
6. **Analyze Results**: Parse execution logs to produce a block-level fix plan (`analysis_plan.json`) and concise analysis summary.
7. **Generate Report**: Synthesize a final report that summarizes results, failures, and next steps.

The workflow supports both interactive mode (user feedback between stages) and full-auto mode. In full-auto mode, the workflow can iterate for a user-specified number of epochs: after `Analyze Results`, it loops back to `Generate Code` for incremental fixes until the epoch budget is exhausted.

## 4. Demonstration Scenario
After installing the TestAgent plugin with `pip install -e .`, the `testagent` CLI is available in the terminal. A new setup requires API credentials. The user edits `~/.testagent_cli/config.json` or uses `testagent config set` to define the model, base URL, and API key. The same configuration file specifies the test command template and optional build or install commands.

The workflow starts from a fully qualified function name and a workspace path, such as `testagent run -f torch.add --workspace ./exam/torch/torch/add`. The agent inspects the target API, produces a requirements note, and generates a structured plan with a small SMOKE set and deferred cases. Interactive mode supports user control with `/next`, `/retry`, and `/goto <stage>`. Full-auto mode advances without prompts and runs the configured test command after code generation.

All artifacts are written under `.testagent/`, including `state.json`, `test_plan.json`, `test_plan.md`, generated test files, execution logs, and coverage outputs. The analysis stage reads log slices, proposes focused fixes, and feeds the next generation cycle. Multi-epoch runs iterate between analysis and code generation for the configured number of epochs. The final report summarizes pass and fail outcomes and records coverage statistics for the target.

## 5. Key Innovations

### 4.1 Multi-Epoch Iterative Feedback
TestAgent implements a deliberate feedback loop in full-auto mode. Each epoch uses analysis artifacts to guide incremental repairs, allowing the system to converge on stable tests rather than relying on a single synthesis pass. Epoch counts are explicit and configurable, providing deterministic control over iteration depth.

### 4.2 Block-Level Incremental Editing
Generated tests are structured using explicit block markers (e.g., `HEADER`, `CASE_*`, `FOOTER`). The workflow enforces a per-block edit limit per epoch: each block can be replaced at most once in a code-generation stage. This prevents destructive rewriting, preserves stability, and encourages minimal diffs aligned with the analysis plan.

### 4.3 Plan-Driven Scope Management
The test plan differentiates **SMOKE_SET** and **DEFERRED_SET** cases. The first epoch is constrained to a small, runnable smoke suite (3–5 cases), while deferred cases are placeholders for later iterations. This staged approach reduces early complexity, improves time-to-first-run, and provides a consistent path for expanding coverage.

### 4.4 Tool-Mediated, Auditable Actions
The LLM does not directly emit code into the response; it uses explicit tools to read, write, and modify files. Tools are registered with JSON schemas and invoked through function calls. This design makes code changes auditable, supports approval gating, and enables deterministic execution within a controlled working directory.

### 4.5 Selective Log Ingestion for Long Contexts
The analysis stage uses on-demand log access (`search` and `part_read`) rather than reading full logs into the prompt. This mitigates prompt-length limitations while still providing sufficient context for diagnosis and planning.

## 6. Implementation Details

### 5.1 State Persistence and Artifact Versioning
Each workflow maintains a `.testagent/state.json` file that stores the current stage, epoch counters, and a history of stage executions. Artifacts are stored under `.testagent/artifacts/<stage>/` with versioned filenames (`vN_*`) and a `current_*` symlink for quick access. This enables deterministic resumption and comparison across iterations.

### 5.2 Tool Registry and Execution
The tool layer includes:
- File discovery and reading (`list_files`, `read_file`, `part_read`, `search`)
- File editing (`write_file`, `replace_in_file`, `replace_block`)
- Command execution (`exec_command`)
- Python introspection (`inspect_python`)

Tool calls are displayed in the UI and can require approval in chat mode. In workflow mode, tools execute in the project root with auto-approval, and missing parameters are explicitly rejected to avoid ambiguous writes.

### 5.3 Execution Pipeline
The execution stage supports optional compile and install commands and a mandatory test command. All commands are configurable via a user config file, including the pytest invocation and environment variables. Logs and exit codes are captured and persisted for downstream analysis.

### 5.4 Human-in-the-Loop Controls
Interactive mode supports explicit commands (`/next`, `/retry`, `/goto <stage>`, `/quit`) and free-form feedback, which is interpreted by a supervisor component to determine the next action. This enables targeted interventions without discarding workflow state.

## 7. Practical Usage
Typical invocation patterns include:
- Interactive workflow: `testagent run -f torch.nn.functional.relu --workspace ./project`
- Full-auto workflow: `testagent run -f torch.add --mode full-auto --epoch 3`
- Resume: `testagent run -f torch.add --resume`

The CLI accepts a workspace (where artifacts are stored) and an optional project root (used as the working directory for code and execution).

## 8. Limitations and Future Work
TestAgent relies on accurate specifications in the planning stage and valid environment configuration for execution. Misconfigured plans (e.g., mismatched file layouts) can prevent test collection. Future work includes automated validation of mock targets, richer static checks before execution, and standardized evaluation across multiple library versions.

## 9. Conclusion
TestAgent demonstrates how a structured, tool-mediated workflow can make LLM-driven test generation more reliable, controllable, and reproducible. By combining staged prompts, artifact-driven planning, and iterative feedback, the system offers a practical template for integrating LLMs into test engineering pipelines while preserving auditability and user control.
