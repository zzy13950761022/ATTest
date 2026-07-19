"""
Stage 4: Generate Code
Creates test code files and build/run scripts.
"""
from ..stage import Stage, StageConfig


class CodeGenStage(Stage):
    """
    Generate complete test code based on requirements and test plan.
    """
    
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        
        self.config = StageConfig(
            name="generate_code",
            display_name="Generate Code",
            description="Generate pytest cases for Python target",
            prompt_template=self._get_prompt_template(),
            input_artifacts=["function_doc.md", "requirements.md", "test_plan.md"],
            output_artifacts=[],  # LLM will write to project path directly
            tools=[
                "inspect_python",
                "list_files",
                "read_file",
                "search",
                "write_file",
                "replace_in_file",
            ],  # Tools for code exploration and generation
            allow_skip=False
        )
    
    def _get_prompt_template(self) -> str:
        return """你正在为 Python 目标 `{target_fqn}` 生成 pytest 单测代码。

当前阶段：Stage 4 - 生成测试代码 → 目标文件 `{test_file_path}`

## 输入
- 必需：`requirements.md`、`test_plan.md`
- 推荐：先用 `read_file` 读取 `function_doc.md`/`requirements.md`/`test_plan.md` 理解约束。
- 迭代修复：如果存在上一轮产物/日志，请先读取并总结问题后再改代码（路径均相对 {project_root}）：
  - 执行日志：`.testagent/artifacts/execute_tests/current_execution_log.txt`
  - 分析报告：`.testagent/artifacts/analyze_results/current_analysis.md`
- 可按需 `inspect_python` 目标获取签名、注解、docstring、源码片段。

## 输出
- 采用 “先骨架、后分块填充” 写入 `{test_file_path}`（相对路径，位于 {project_root} 下）。
  - 第 1 步：用 `write_file` 写入精简骨架，只包含 import、固定 helper/fixture、测试类/函数声明和占位符（占位符格式：`# ==== BLOCK:CASE_XX ====`, 唯一且易检索），行数尽量 < 200。
  - 第 2 步：针对每个占位符，用 `replace_in_file` 逐块填充完整测试逻辑；每块控制在 ~300 行或 < 12k 字符，`search` 必须精确包含占位符文本。
  - 第 3 步：填充后用 `read_file` 检查关键用例/断言/占位符已替换，确认文件未被截断或意外缩短。
  - 禁止在一次 `write_file` 中写入完整大文件；禁止清空/覆盖已填充块。

## 代码要求
1. 使用 `pytest`，命名规范：`test_*.py`、`test_*` 函数。
2. 覆盖 test_plan 中的所有测试用例，补充 requirements 里的约束（shape/dtype/异常）。
3. 构造输入时固定随机种子，避免依赖外部资源；如需外部依赖，使用 `unittest.mock`/`monkeypatch` stub。
4. 对返回值/副作用/异常做明确断言；浮点比较使用合适的容差。
5. 若目标是类/方法，包含实例化逻辑或使用简化的假实现/fixture。
6. 兼容 CPU 环境，避免 GPU/分布式等重度依赖，除非需求明确。

## 建议结构
```python
import math
import pytest
from package.module import target  # 根据 {target_fqn} 填写

def test_happy_path():
    ...

@pytest.mark.parametrize(...)
def test_edge_case(...):
    ...

def test_invalid_inputs(...):
    with pytest.raises(...):
        ...
```

在文件头部添加必要的 import（含目标函数/类），保持代码可直接运行。
只通过上述多步工具写入文件，不要在对话中粘贴源代码。
"""
    
    def get_config(self) -> StageConfig:
        return self.config
