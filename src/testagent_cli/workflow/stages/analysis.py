"""
Stage 6: Analyze Results
Analyzes test execution logs to identify issues and suggest fixes.
"""
from ..stage import Stage, StageConfig


class AnalysisStage(Stage):
    """
    Analyze test execution results, identify failures, and suggest fixes.
    """
    
    def __init__(self, llm, tool_runner):
        super().__init__(llm, tool_runner)
        self.config = StageConfig(
            name="analyze_results",
            display_name="Analyze Results",
            description="Parse pytest logs and identify issues",
            prompt_template=self._get_prompt_template(),
            input_artifacts=["exit_code.txt"],
            output_artifacts=["analysis.md", "analysis_plan.json"],
            tools=["write_file", "read_file", "part_read", "list_files", "search"],
            allow_skip=False
        )
    
    def _get_prompt_template(self) -> str:
        return """你是测试分析助手，负责解读 pytest 运行日志并输出“块级修复计划”。

阶段：Stage 6 - 结果分析 → 输出 `analysis.md` 与 `analysis_plan.json`

## 输入
- 执行日志路径：`.testagent/artifacts/execute_tests/current_execution_log.txt`
- 退出码：{exit_code_txt}

## 读取策略（必须遵守）
1) **不要**把整段日志直接读入 prompt。
2) 先用 `search` 定位关键片段（如 FAILED/ERROR/AssertionError/Traceback）。
3) 再用 `part_read` 读取相关上下文（只取必要片段）。
4) 仅在日志非常短时才允许 `read_file` 全量读取。

## 输出要求（必须遵守）
1) **只输出块级修复计划**，避免长篇分析文本。
2) 每轮最多给出 1~3 个待修改 BLOCK（其余标记为 deferred）。
3) 失败用例必须映射到 BLOCK_ID（优先参考 `test_plan.json` 中的 block_id；公共依赖/导入/fixture 用 `HEADER`；清理/收尾用 `FOOTER`）。
4) 若发现覆盖率缺口，可新增 BLOCK（action=`add_case`），但仍受 1~3 个限制。
5) 如可判断与上一轮失败集合完全重复，可设置 `stop_recommended=true` 并给出 `stop_reason`；若仅错误类型重复，不要 stop_recommended，改为将对应 BLOCK 标记为 deferred 并在 reason 中注明“错误类型重复，跳过该块”。

## analysis_plan.json（机器可读）
写成严格 JSON，字段如下：
```
{{
  "status": "成功|未完全通过|失败",
  "passed": <int>,
  "failed": <int>,
  "errors": <int>,
  "collection_errors": <bool>,
  "block_limit": 3,
  "failures": [
    {{
      "test": "<node id>",
      "block_id": "<BLOCK_ID>",
      "error_type": "<AssertionError/TypeError/...>",
      "action": "rewrite_block|adjust_assertion|fix_dependency|add_case|mark_xfail",
      "note": "<short reason>"
    }}
  ],
  "deferred": [
    {{"test": "<node id>", "reason": "<short>"}}
  ],
  "stop_recommended": <bool>,
  "stop_reason": "<short>"
}}
```

## analysis.md（简洁可读）
仅包含：
- 状态与通过/失败统计
- 待修复 BLOCK 列表（<=3，含 action 与 error_type）
- stop_recommended/stop_reason（如为 true）

使用 `write_file` 写入 `analysis_plan.json` 与 `analysis.md`，对话中只给一句摘要。"""
    
    def get_config(self) -> StageConfig:
        return self.config
