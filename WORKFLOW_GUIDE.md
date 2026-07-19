# TestAgent-CLI 完整使用指南（Python 版）

适用于 Python API/算子（PyTorch、TensorFlow 等）的 7 阶段测试生成工作流。

## 📚 目录
1. [快速开始](#快速开始)
2. [Workflow 模式详解](#workflow-模式详解)
3. [配置自定义命令](#配置自定义命令)
4. [测试与调试](#测试与调试)
5. [扩展与定制](#扩展与定制)
6. [常见问题](#常见问题)

---

## 快速开始

### 安装
```bash
cd TestAgent-CLI-main
pip install -e .
```

### 基础使用
#### 模式 1: 聊天模式
```bash
testagent chat --workspace ~/my-project
```

#### 模式 2: 工作流模式
```bash
# 交互式（推荐）
testagent run -f torch.nn.functional.relu --workspace ~/my-project

# 全自动
testagent run -f torch.add --mode full-auto

# 恢复中断
testagent run -f torch.add --resume
```

---

## Workflow 模式详解

### 7 阶段概览
```
1. understand_function    → 分析 Python 目标
2. generate_requirements  → 生成需求
3. design_test_plan       → 设计测试计划
4. generate_code          → 生成 pytest 代码
5. execute_tests          → 运行 pytest
6. analyze_results        → 分析结果
7. generate_report        → 生成报告
```

### 交互模式示例
```
╔═══════════════════════════════════════════════╗
║   TestAgent Workflow - torch.add (python)     ║
╠═══════════════════════════════════════════════╣
║ [▶] 1. Understand Function ← Current Stage    ║
║ [ ] 2. Generate Requirements                  ║
║ [ ] 3. Design Test Plan                       ║
║ [ ] 4. Generate Code                          ║
║ [ ] 5. Execute Tests                          ║
║ [ ] 6. Analyze Results                        ║
║ [ ] 7. Generate Report                        ║
╚═══════════════════════════════════════════════╝

⚙️  Executing: Understand Function
...
```
- 按 Enter 继续
- `/regenerate` 重新生成当前阶段
- `/retry 需要覆盖空张量` 带反馈重试
- `/goto generate_requirements` 跳转阶段

### 工作目录结构
```
workspace/
├── .testagent/
│   ├── state.json
│   └── artifacts/
│       ├── understand_function/current_function_doc.md
│       ├── generate_requirements/current_requirements.md
│       ├── design_test_plan/current_test_plan.md
│       ├── generate_code/current_tests_test_<target_slug>.py
│       └── ...
└── tests/test_<target_slug>.py   # 生成的 pytest 文件
```

---

## 配置自定义命令

编辑 `~/.testagent_cli/config.json`：
```json
{
  "project": {
    "root": ".",
    "test_file_template": "tests/test_{target_slug}.py"
  },
  "commands": {
    "compile": "",
    "install": "",
    "run_test": "PYTHONPATH={project_root}:$PYTHONPATH pytest -q {test_file_path}"
  }
}
```
可用变量：`{target}` `{target_slug}` `{project_root}` `{test_file_path}`。

常见调整：
- 只跑部分用例：`pytest -q {test_file_path} -k cpu`
- 增加覆盖率：`pytest -q {test_file_path} --cov=your_pkg`

---

## 测试与调试

### 运行自带测试
```bash
pytest test_workflow_e2e.py -q
pytest test_smoke.py -q
```

### 查看状态与产物
```bash
cat workspace/.testagent/state.json
ls workspace/.testagent/artifacts
```

### 单阶段调试示例
```python
# debug_stage.py
from testagent_cli.workflow import WorkflowState
from testagent_cli.workflow.stages import RequirementsStage
from testagent_cli.tools import build_default_registry, ToolRunner
from testagent_cli.llm import LLMClient

state = WorkflowState("/tmp/ws", "relu", "python", target="torch.nn.functional.relu")
state.save_artifact("function_doc.md", "# mock doc")
llm = LLMClient(model="deepseek-chat")
runner = ToolRunner(build_default_registry())
result = RequirementsStage(llm, runner).execute(state)
print(result.success, result.error)
```

---

## 扩展与定制

- 修改 Stage 提示词：编辑 `src/testagent_cli/workflow/stages/*.py` 的 `_get_prompt_template()`
- 添加新 Tool：在 `tools/builtin.py` 创建类，并在 `tools/runner.py` 注册
- 添加新 Stage：新增 `workflow/stages/your_stage.py`，在 `stages/__init__.py` 注册，并在 `workflow/engine.py` 添加到 `STAGE_NAMES`

---

## 常见问题

**Q: 生成的 pytest 路径在哪？**  
A: 默认 `tests/test_<target_slug>.py`，可通过 `project.test_file_template` 配置。

**Q: 如何跳过某阶段？**  
A: `/goto <stage>` 跳转到目标阶段，或 `/next` 直接继续。

**Q: 日志/产物在哪？**  
A: `.testagent/state.json` 和 `.testagent/artifacts/`。

**Q: 运行命令想自定义？**  
A: 设置 `commands.run_test`，模板变量见上。

祝使用顺利！ 🎉
