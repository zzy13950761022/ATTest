"""
Stage 3: Design Test Plan
Creates a concrete test plan with specific test cases.
"""
from ..stage import Stage, StageConfig


class TestPlanStage(Stage):
    """
    Design specific test cases based on requirements.
    """
    
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="design_test_plan",
            display_name="Design Test Plan",
            description="Plan specific test cases and scenarios",
            prompt_template=self._get_prompt_template(),
            input_artifacts=["function_doc.md", "requirements.md"],
            output_artifacts=["test_plan.md", "test_plan.json"],
            tools=["read_file", "write_file"],
            allow_skip=False
        )
    
    def _get_prompt_template(self) -> str:
        return """为目标 `{target_fqn}` 设计“生成规格书”型测试计划（pytest 视角）。要求：test_plan 是后续 TC 生成的唯一依据，首轮只产出少量可运行用例，便于迭代补齐。

阶段：Stage 3 - 设计测试计划 → 输出 `test_plan.md` 与 `test_plan.json`

## 步骤
1) 使用 `read_file` 读取 `function_doc.md`、`requirements.md`。
2) 生成 `test_plan.json`（机器可读、唯一规格源），并生成 `test_plan.md`（简短摘要 + 引用规格）。

## 关键约束（必须遵守）
1) **SMOKE_SET 优先**：首轮只生成 3-5 个核心用例，保证最小可运行。
2) **Low/Medium 不得新增独立 CASE**：必须作为已有 High CASE 的参数维度扩展。
3) **断言分级**：每个 CASE 必须声明 weak/strong 断言列表，首轮只用 weak。
4) **预算控制**：每个 CASE 必须声明 size / max_lines / max_params / is_parametrized / requires_mock。
5) **迭代策略固定**：在规格中写死首轮/后续/最后一轮的生成策略。
6) **单文件默认**：默认输出单文件计划，不要生成 `groups`/`active_group_order`/`test_files.groups`。
7) **BLOCK_ID 映射**：每个 TC 必须与 CASE_XX 一一对应；BLOCK_ID 稳定不可变。
8) **Mock 目标明确**：当 `requires_mock=true` 时，必须填写 `mock_targets`（完整导入路径列表），并确保与 requirements 中的 mock 约束一致。

## `test_plan.json` 结构（严格 JSON）
```json
{{
  "plan_version": 2,
  "target": "{target_fqn}",
  "block_rules": {{
    "header_block": "HEADER",
    "footer_block": "FOOTER",
    "case_prefix": "CASE_",
    "case_format": "CASE_01"
  }},
  "iteration_strategy": {{
    "round1": {{"include": "SMOKE_SET", "assert_level": "weak", "max_blocks": 5}},
    "roundN": {{"only_fix_failed_blocks": true, "block_limit": 3, "promote_deferred": true}},
    "final": {{"enable_strong_asserts": true, "coverage_optional": true}}
  }},
  "test_files": {{
    "default": "tests/test_{target_slug}.py",
    "all_pattern": "tests/test_{target_slug}.py"
  }},
  "cases": [
    {{
      "tc_id": "TC-01",
      "block_id": "CASE_01",
      "name": "核心路径",
      "priority": "High",
      "param_matrix": [
        {{"dtype": "float32", "device": "cpu", "shape": [2, 2], "flags": []}}
      ],
      "asserts": {{
        "weak": ["shape", "dtype", "finite", "basic_property"],
        "strong": ["approx_equal", "orthogonality"]
      }},
      "oracle": "torch.linalg.eigh",
      "assertion_level": "weak",
      "size": "S",
      "max_lines": 80,
      "max_params": 6,
      "is_parametrized": true,
      "requires_mock": false,
      "mock_targets": ["torch.nn.parallel.scatter_gather.scatter_kwargs"]
    }}
  ],
  "param_extensions": [
    {{
      "base_block_id": "CASE_01",
      "priority": "Medium",
      "params": {{"dtype": "float64", "device": "cpu", "shape": [4, 4], "flags": ["edge"]}},
      "note": "作为参数扩展"
    }}
  ],
  "smoke_set": ["CASE_01", "CASE_02", "CASE_03"],
  "deferred_set": ["CASE_04", "CASE_05", "CASE_06"]
}}
```

## `test_plan.md` 结构（简短摘要）

```markdown
# {target_fqn} 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures
- 随机性处理：固定随机种子/控制 RNG

## 2. 生成规格摘要（来自 test_plan.json）
- SMOKE_SET: 列出 CASE_XX
- DEFERRED_SET: 列出 CASE_XX
- 测试文件路径（单文件）
- 断言分级策略（weak/strong）
- 预算策略（size/max_lines/max_params）

## 3. 数据与边界
- 正常数据集与随机生成策略（短句）
- 边界值/极端形状/空输入（每条不超过 15 字）
- 负例与异常场景列表（只列标题）

## 4. 覆盖映射
- 每个 TC 对应的需求/约束（可用表格或短列表）
- 尚未覆盖的风险点（只列关键风险）
```

**只用 `write_file` 写入 `test_plan.json` 与 `test_plan.md`**，不要在对话中粘贴全文。
{user_feedback}
"""
    
    def get_config(self) -> StageConfig:
        return self.config
