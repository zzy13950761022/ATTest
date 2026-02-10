# Workflow Module - Phase 2 Implementation

## Overview

The workflow module provides a stateful, stage-based test generation engine with intelligent feedback processing.

## Architecture

```
workflow/
├── state.py         # Workflow state management with persistence
├── stage.py         # Base class for all stages (sub-agents)
├── engine.py        # Main workflow orchestrator
├── display.py       # User interface and progress visualization
├── supervisor.py    # Intelligent feedback interpreter
└── stages/          # 7 workflow stages
    ├── understand.py       # Stage 1: Analyze operator
    ├── requirements.py     # Stage 2: Generate test requirements
    ├── planning.py         # Stage 3: Design test plan
    ├── codegen.py          # Stage 4: Generate test code
    ├── execution.py        # Stage 5: Run tests
    ├── analysis.py         # Stage 6: Analyze results
    └── report.py           # Stage 7: Generate final report
```

## Quick Start

```python
from testagent_cli.workflow import WorkflowEngine
from testagent_cli.llm import LLMClient

# Initialize engine
llm = LLMClient(model="deepseek-chat", ...)
engine = WorkflowEngine(
    llm=llm,
    workspace="/path/to/workspace",
    op="conv2d",
    arch="x86"
)

# Run workflow
engine.run(mode="interactive")  # or "full-auto"
```

## CLI Usage

```bash
# Interactive mode (default)
testagent test --op conv2d --arch x86

# Full auto mode
testagent test --op matmul --arch arm --mode full-auto

# Resume interrupted workflow
testagent test --op add --arch x86 --resume
```

## Features

### 7-Stage Pipeline
1. **Understand Function** - Analyze operator semantics
2. **Generate Requirements** - Define test requirements
3. **Design Test Plan** - Create specific test cases
4. **Generate Code** - Write test code and scripts
5. **Execute Tests** - Run and capture results
6. **Analyze Results** - Identify issues and root causes
7. **Generate Report** - Produce comprehensive report

### Intelligent Feedback
- **Special Commands**: `/next`, `/retry`, `/goto <stage>`, `/quit`
- **Natural Language**: "需要增加边界测试"
- **LLM-powered Intent Classification**: Understands user intent

### State Management
- **Persistence**: Save/load workflow state
- **Recovery**: Resume after interruption
- **Artifact Versioning**: Track changes across regenerations
- **History Tracking**: Record all stage executions and feedback

### Dual Modes
- **Interactive**: User reviews and provides feedback after each stage
- **Full-Auto**: Execute all stages without stopping

## Implementation Status

✅ **Milestone 1**: Core Framework (state, stage, engine, display)
✅ **Milestone 2**: 7 Workflow Stages  
✅ **Milestone 3**: Supervisor Agent
✅ **Milestone 4**: CLI Integration & Testing

## Testing

```bash
# Run integration tests
pytest test_workflow_e2e.py -v

# Quick smoke test
python test_smoke.py
```

All tests passing (5/5 integration tests + smoke test).

## Next Steps

1. Test with real operators and LLM
2. Customize build scripts for your environment
3. Provide feedback on stage prompts
4. Report any issues or enhancement requests

For detailed documentation, see [walkthrough.md](file:///Users/mac/.gemini/antigravity/brain/d2a649c9-7a89-48c4-8bc7-284469311957/walkthrough.md).
