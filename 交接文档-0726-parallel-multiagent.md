# ATTest 交接文档（2026-07-26 补充：多 Agent 并行 + 闭环 + 混合 case 规划）

> 本文档承接 `交接文档-0726.md`，记录在其之后新增的一次**较大重构**：借鉴 TestAgent 论文的多 agent 模式，实现「并行生成 + planner/reviewer 闭环 + LLM 混合 case 规划」，目标是把单算子一轮耗时从 EPOCH=4 的 ~4.3h（sign 甚至 10.3h）大幅压低。

---

## 一、这次改了什么（一句话）

把 `AscendGenerationAgentLoopStage` 的**单 300-turn 串行 session** 拆成 **op_api / op_host 两个 generator agent 用线程池并行生成**（独立 build 目录隔离编译，join 后合并覆盖率）；并让 **reviewer（analysis）在覆盖率平台期回灌 planner**，planner 用一次 LLM 调用针对未覆盖分支生成针对性 case（首轮仍走规则生成保速度）。

---

## 二、动机

- 交接文档 0726 记录：EPOCH=4 平均 4.3h/算子，sign 花 10.3h 反复修 op_host；根因是两层被塞进一个串行 loop，且 planner 只在开头跑一次，reviewer 发现平台期时无法回灌补充新需求。
- 代码探索确认：**op_api 与 op_host 在生成测试上无真实依赖**——被测源码目录、`*_attest.cpp`、各自目录下的 CMakeLists、编译 flag（`--opapi` vs `--ophost`）全分离。prompt 里的 "op_api first" 只是单 agent 的优先级建议，不是硬依赖。
- 论文 RQ2 消融：LLM 失去细粒度代码视野时行覆盖掉 22%、分支掉 24% → LLM 生成 case 显著优于规则枚举，但成本高，故做**混合**。

---

## 三、已确认的设计决策

1. **分两阶段**：阶段一并行（优先），阶段二闭环。
2. **并发 = ThreadPoolExecutor**：每 agent 一线程，复用现有同步 `LLMClient`（LLM 调用 IO-bound，阻塞时释放 GIL，真并行）。
3. **build 隔离 = 共享源码 + 独立 build 目录**：每层 `BUILD_PATH=build_generated_{layer}_{op}`，复用已有 `_patch_build_sh_for_isolation`。
4. **reviewer = 覆盖评估 + 回灌 planner**（augment 针对性 case），不做 per-case accept/discard。
5. **只并行 LLM 生成，per-layer 覆盖率编译在主线程串行**——彻底消除 build.sh patch/restore 并发竞争，且几乎零墙钟代价（慢的是生成不是覆盖率提取）。

---

## 四、改动清单（未 commit 前为 4 文件 +633/-47）

### 4.1 `src/attest_cli/cli.py`
- `run_ascend_ut` 新增 `--workers`（默认 1）；`_launch_ascend_workflow` 加 `workers` 参数并传入 `WorkflowEngine`。

### 4.2 `src/attest_cli/workflow/engine.py`
- `__init__` 加 `workers` 参数；`self.state.workers = max(1, getattr(self.state, "workers", workers))`（仿 epoch 持久化，`--resume` 保留）。
- `_register_stages` 后：若 `generate_code` stage 有 `set_workers` 则注入。
- **epoch 循环**（原只跳 `generate_code`）：新增分支——若刚写的 `analysis_plan.json` 有 `replan_recommended` 且存在 `design_test_plan` stage，则跳回 `design_test_plan`（planner 重规划），否则维持跳 `generate_code`。

### 4.3 `src/attest_cli/workflow/state.py`
- `persist()` 写入 `"workers"`；`load()` 读回 `state.workers`（`--resume` 真正保留 workers）。

### 4.4 `src/attest_cli/workflow/ascend_stages.py`（主改动）

**模块级**
- `append_message` 改为线程安全包装（`_APPEND_LOCK` 保护 session JSONL 写入）；原 `session.append_message` 以 `_raw_append_message` 导入。新增 `import threading / copy / concurrent.futures`。

**`AscendGenerationAgentLoopStage`（并行生成）**
- `set_workers` / `self.workers`。
- `_layer_files(plan, layer)`：取某层的 cpp 文件（排除 cmake）。
- `_per_layer_turn_limit()`：`max(60, 300 // workers)`，保持总 turn 预算恒定。
- `_isolated_build_prefix(root, layer, op)`：生成 `BUILD_PATH=.../build_generated_{layer}_{op} BUILD_OUT_PATH=...` 前缀，烤进 prompt 里的编译/覆盖命令（因连续 loop 是 LLM 自己发 exec_command，框架前缀不生效）。
- `_build_single_layer_prompt(...)`：单层作用域 prompt（只列该层文件 + 该层带前缀的编译命令 + augment directive）。
- `_run_layer_session(...)`：worker 线程体，跑一次 `_run_llm_session`（**不** save_artifact、**不** patch build.sh）。
- `_collect_parallel_coverage(...)`：join 后主线程**逐层串行**跑 `AscendExecutionStage._run_for_root(label=f"generated_{layer}")`，合并 `per_layer` → 按 `_build_run_summary` 生成 `coverage_summary.json` 等 artifact（schema 与串行完全一致，`generate_report` 无需改）。
- **`execute` 改写**：`workers>1 且 层数>1` → 并行分支（主线程 up-front patch build.sh 一次 → ThreadPoolExecutor 每层一 agent → 主线程存 manifest + `_collect_parallel_coverage` → `finally` 恢复 build.sh → analysis）；否则 → **原单 session 路径完全不变（向后兼容）**。

**`AscendAnalysisStage`（reviewer）**
- `_augment_uncovered_digest(uncovered_analysis, layers)`：抽取平台层的未覆盖行精简摘要。
- `execute` 平台期分支：`no_improvement_rounds >= patience` 时，若 `augment_rounds < 1` 且有层低于阈值 → 发 `augment_request`（layers + uncovered 摘要）+ `replan_recommended=True`（**不**设 `stop_recommended`，故 `_check_auto_stop` 不会误停），`augment_rounds+1`；否则维持原来的 `stop_recommended`。写入 `analysis_plan.json`。

**`AscendTestPlanStage`（planner，混合 case 规划）**
- `execute` 开头 load `analysis_plan.json`，epoch>1 且有 `augment_request` 时记 `augment_request`，并把它作为 `plan["augment_directive"]` 传给下游 generator prompt。
- 规则 `_build_cases` 后，若 augment：调 `_llm_augment_cases` 生成针对性 case **追加**到规则 case（延续 `CASE_NN`/`TC-NN`，schema 一致，标 `origin: llm_augment`；High→smoke_set，其余→deferred_set）。
- `_llm_augment_cases(...)`：一次 `self.llm.chat`，输入=未覆盖摘要+源码可用文件，输出=3-8 个针对性 case；非法 `file_id` 丢弃；**任何失败（异常/垃圾/空）返回 `[]` 回退纯规则**。
- `_extract_json_array(text)`：从 ```json 数组``` 或裸数组解析。

---

## 五、数据流（阶段一并行 + 阶段二闭环）

```
epoch N (workers=2):
  design_test_plan (planner)
    - 规则生成 ~40 case（首轮）
    - 若 epoch>1 且有 augment_request: LLM 追加针对性 case → plan.augment_directive
  generate_code (AscendGenerationAgentLoopStage)
    - 主线程 patch build.sh 一次
    - ThreadPoolExecutor:
        agent[op_api]  → _run_llm_session（BUILD_PATH=build_generated_op_api_{op}）
        agent[op_host] → _run_llm_session（BUILD_PATH=build_generated_op_host_{op}）
    - join → 主线程 _collect_parallel_coverage（逐层串行 _run_for_root）→ coverage_summary.json
    - finally 恢复 build.sh
    - AscendAnalysisStage.execute（reviewer）
        - 覆盖达标 → stop
        - 平台期 & 有层欠覆盖 & augment_rounds<1 → augment_request + replan_recommended
  engine epoch 边界:
    - replan_recommended → jump design_test_plan（planner 重规划，带 augment）
    - 否则 → jump generate_code
```

---

## 六、验证状态

### 已验证（本机 macOS，Python 3.9）
- 4 个改动文件全部 `py_compile` 通过。
- **纯逻辑单测通过**（用 AST 抽取函数 + 打桩独立跑）：
  - `_layer_files` / `_isolated_build_prefix` / `_per_layer_turn_limit` / `_augment_uncovered_digest`
  - `_llm_augment_cases`（打桩 LLM）：正常解析、编号延续 `CASE_13/14`、非法 file_id 丢弃、三种失败模式（异常/垃圾/空）全部回退 `[]`
  - `_extract_json_array`：fenced / 裸数组 / 无 json / 坏 json

### 未验证——**必须在目标机跑**（本机是 Python 3.9，项目要求 ≥3.11；且需真实 LLM + Ascend build 环境）
1. `python test_smoke.py`（本机因 legacy 包 `str | Path` 语法无法跑，非本次改动所致）。
2. `attest ascend-ut run --help` 应显示 `--workers`。
3. **向后兼容**：单算子默认 `--workers 1`，`--profile ascend_ut_continuous -m full-auto -e 1`，`timeout 120` → 应与旧单 session 行为一致，`coverage_summary.json` 合法。
4. **并行**：同算子 `--workers 2 -m full-auto`，`timeout 120` → 确认
   - `build_generated_op_api_*` / `build_generated_op_host_*` 目录创建后被清理
   - `generated_op_api_*.lcov.info` 与 `generated_op_host_*.lcov.info` 两 artifact 都在
   - `git diff build.sh` 为空（已恢复）、build.sh 未损坏
5. **合并正确性**：同算子 diff 并行 vs 串行 `coverage_summary.json["per_layer"]` → per-layer `line_coverage` 噪声内一致；`overall`=有效层均值；schema key `per_layer`/`overall`/`coverage_valid`/`thresholds` 齐全。
6. **阶段二闭环**：强制平台期（小 epoch + 低覆盖算子）→ `analysis_plan.json` 出现 `replan_recommended`+`augment_request`，engine 日志打印跳 `design_test_plan`，下 epoch `test_plan.json` 出现 `origin: llm_augment` 的新 case。

---

## 七、已知风险 / 注意事项

| 风险 | 说明 | 缓解 |
|---|---|---|
| **augment 新 case 无预置 block 骨架** | `_ensure_skeleton` 在 epoch>1 遇已存在测试文件会跳过，不会为新 `CASE_NN` 铺 placeholder marker | generator 收到完整 plan（含新 case）+ 明确 "ADD new TEST_F" 指令 + `append_to_file` 工具，会自行追加；block marker 是给 analysis 溯源用、非硬门槛。**目标机需确认 augment case 确实被 generator 落地** |
| **`_run_for_root` 自身也 patch build.sh** | 并行路径主线程已 up-front patch；`_collect_parallel_coverage` 串行调 `_run_for_root` 时它又 backup(已patched)→restore(patched)，我的 finally 再从 `.sh.parallel.bak` 恢复原始 | patch 幂等（`${BUILD_PATH:-` 守卫），双 patch 无害；backup 文件名不冲突（`.sh.v9.bak` vs `.sh.parallel.bak`）；顺序执行无竞争 |
| **workers>1 的 turn 预算** | 每层 `300//workers` turn，2 层各 150 | 换墙钟不换总预算；若发现单层 150 turn 不够可调 `_per_layer_turn_limit` |
| **只对 continuous profile 生效** | 并行分支在 `AscendGenerationAgentLoopStage`；`ascend_ut`（7-stage）走各自独立 stage，`--workers` 对它无并行效果 | 并行需 `--profile ascend_ut_continuous --workers 2` |
| **LLM augment 依赖 uncovered_code.json** | 若 analysis 未采集到未覆盖行，augment digest 为空，`_llm_augment_cases` 回退 [] | 与现有 uncovered 采集共用逻辑，无新依赖 |
| **v14 state.json 兼容** | 旧 state 无 `workers` 字段 | `getattr(self.state, "workers", workers)` + load `data.get("workers", 1)`，无需迁移 |

---

## 八、如何使用

```bash
# 并行（推荐，2 路 = op_api / op_host 各一 agent）+ 闭环
attest ascend-ut run \
  --op-path math/div -p <repo_root> \
  --profile ascend_ut_continuous \
  -m full-auto -e 2 \
  --workers 2

# 向后兼容（旧行为，单 session）
attest ascend-ut run --op-path math/div -p <repo_root> \
  --profile ascend_ut_continuous -m full-auto -e 2 --workers 1
```

批量脚本：在 `run_attest_ut_batch_v14_continuous_resumed.sh` 的 attest 调用里加 `--workers 2` 即可。**建议先单算子验证（第六节 3-6 项）再全量跑。**

---

## 九、下一步建议（给目标机）

1. **P0 验证**：按第六节 3-6 项逐条跑，重点是并行的 build 隔离 + 覆盖率合并正确性（这是 C 算子的最大风险点）。
2. **P1 提速验收**：单算子并行 vs 串行耗时对比，目标 op_host/op_api 墙钟从串行的 sum 降到 ~max（约 2x 加速），覆盖率不掉。
3. **P2 闭环验收**：观察 op_host 0% 类算子（div/mod/floor_div）在 augment 后 op_host 覆盖是否上升。
4. **P3 若并行度不够**：可考虑层内 CASE 再拆分（当前只到 layer 粒度），但会引入同文件写入/CMakeLists 冲突，需额外加锁——非必要不做。

---

## 十、真源

- 远程：`github.com:zzy13950761022/ATTest.git` 分支 `ascend`
- 本次 commit：见 `git log`（本文件同批推送）
- 参考论文：TestAgent（Multi-Agent LLM Collaboration for Enhancing Unit Test Generation），仓库根目录 PDF
- 上份交接：`交接文档-0726.md`

*最后更新：2026-07-26。新增多 agent 并行（op_api/op_host 线程池）+ reviewer→planner 闭环 + LLM 混合 case 规划。纯逻辑已单测通过，端到端待目标机（Python 3.11 + Ascend 环境）验证。*
