# ATTest-CLI 快速参考

## 🚀 常用命令

```bash
# 默认交互模式（聊天）
attest [--workspace DIR] [--auto-approve]

# 聊天模式（显式）
attest chat [--workspace DIR] [--auto-approve]

# 工作流模式（交互式）- 短参数形式
attest run -f package.module:function [--workspace DIR] [--project-root DIR]

# 工作流模式（交互式）- 长参数形式
attest run --func package.module:function [--workspace DIR]

# 工作流模式（全自动）
attest run -f package.module:function --mode full-auto

# 全自动 + 多轮迭代（如 3 轮）
attest run -f package.module:function --mode full-auto --epoch 3

# 恢复中断的工作流
attest run -f package.module:function --resume

# 配置管理
attest config list
attest config set KEY VALUE
attest config get KEY

# 会话管理
attest sessions list
attest sessions clear <session_id>
```

## 📝 Workflow 交互命令

在每个阶段完成后，你可以使用：

| 命令 | 说明 |
|------|------|
| `Enter` | 批准并继续下一阶段 |
| `/next` | 同上 |
| `/regenerate` | 重新生成当前阶段 |
| `/retry` | 重新生成（可选带反馈，如 `/retry 需要覆盖空张量`） |
| `/goto <stage>` | 跳转到指定阶段（如 `/goto generate_code`） |
| `/status` | 查看工作流状态 |
| `/help` | 显示帮助信息 |
| `/quit` | 退出工作流 |
| `<自然语言>` | 智能理解反馈（SupervisorAgent 自动解析） |

## 🔧 配置自定义构建命令

编辑 `~/.attest_cli/config.json`：

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

可用变量：
- `{target}` / `{target_slug}` - 目标函数 FQN 及其 slug
- `{project_root}` - 项目根目录
- `{test_file_path}` - 生成的 pytest 文件路径

## 🎯 Workflow 7 阶段

```
1. understand_function    → 分析 Python 目标
2. generate_requirements  → 生成需求
3. design_test_plan       → 设计测试计划
4. generate_code          → 生成 pytest 代码
5. execute_tests          → 运行 pytest
6. analyze_results        → 分析结果
7. generate_report        → 生成报告
```

## 📂 产物位置

```
workspace/
├── .attest/
│   ├── state.json                    # 工作流状态
│   ├── artifacts/                    # 各阶段产物（带版本控制）
│   │   ├── understand_function/
│   │   │   ├── current_function_doc.md    # 当前版本符号链接
│   │   │   └── v1_function_doc.md         # 版本化存储
│   │   ├── generate_requirements/
│   │   │   ├── current_requirements.md
│   │   │   └── v1_requirements.md
│   │   └── ...
│   └── logs/                         # 日志目录
├── tests/test_<target_slug>.py       # 生成的 pytest 文件
└── （可选）其他项目文件
```

## 🛠️ 快速定制

### 修改 Stage Prompt

```bash
vi src/attest_cli/workflow/stages/requirements.py
```

编辑 `_get_prompt_template()` 方法。

### 添加新 Tool

1. 在 `src/attest_cli/tools/builtin.py` 添加类
2. 在 `src/attest_cli/tools/runner.py` 注册
3. 在 Stage 的 `tools` 列表中使用

### 调试

```bash
# 查看状态
cat workspace/.attest/state.json

# 查看产物
ls workspace/.attest/artifacts/

# 单元测试
pytest test_workflow_e2e.py -q
pytest test_smoke.py -q
```

## ⚡ 示例

### 标准使用

```bash
attest run -f torch.nn.functional.relu --workspace ~/my-proj
```

### 覆盖自定义 pytest 命令

```bash
# 1. 配置
attest config set commands.run_test "PYTHONPATH={project_root}:$PYTHONPATH pytest -q {test_file_path} -k gpu"

# 2. 运行
attest run -f torch.add --mode full-auto
```

### 中途修改需求

```
阶段 2 完成后：
> 需求太简单，需要增加并发测试和性能测试

阶段继续，需求会被重新生成
```

---

详细文档请参考 [WORKFLOW_GUIDE.md](./WORKFLOW_GUIDE.md)
