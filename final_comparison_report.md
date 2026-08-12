# ATTest 实验对比报告 (最终版)

**生成时间**: 2026-08-12
**算子数量**: 45
**覆盖率指标**: op_host/op_api 层 LINE coverage, overall = (host + api) / 2 (N/A 按 0 计)

---

## 1. 方案说明

| 方案 | 标识 | 说明 |
|------|------|------|
| B2 | opencode plain | opencode CLI 无 skill 指导，纯 LLM 能力 |
| B3 | opencode + skill | opencode CLI + Ascend C skill 指导 |
| ATTest | v15 E3-fixed-v2 | ATTest 自动化 workflow, epoch=3, timeout=18000s |

**模型版本**:
- Qwen: qwen3-coder-plus
- DS: deepseek-v4-pro
- GLM: glm-5.2

---

## 2. 总体对比

| 模型 | 方案 | Valid | Avg host | Avg api | Avg overall | Wall h |
|------|------|------:|---------:|--------:|------------:|-------:|
| qwen  | B2      |   44/45 |    51.1% |   70.8% |       61.0% | 28h |
| qwen  | B3      |   45/45 |    55.9% |   71.2% |       63.6% | 27h |
| qwen  | ATTest  |   28/45 |    45.5% |   31.6% |       38.6% | 12h |

| ds    | B2      |   43/45 |    42.2% |   71.2% |       56.7% | 27h |
| ds    | B3      |   44/45 |    44.6% |   70.2% |       57.4% | 21h |
| ds    | ATTest  |   36/45 |    53.3% |   58.9% |       56.1% | 69h |

| glm   | B2      |   41/45 |    46.2% |   71.8% |       59.0% | 61h |
| glm   | B3      |   43/45 |    52.5% |   72.7% |       62.6% | 54h |
| glm   | ATTest  |   31/45 |    45.5% |   39.8% |       42.7% | 112h |

---

## 3. 逐算子对比

| 算子 | Qwen B2 | Qwen B3 | Qwen ATTest | DS B2 | DS B3 | DS ATTest | GLM B2 | GLM B3 | GLM ATTest |
|------|--------:|--------:|------------:|------:|------:|----------:|-------:|-------:|-----------:|
| abs                    | 84% | 44% | 44% | 47% | 48% | 48% | 47% | 48% | 0% |
| add                    | 84% | 84% | 81% | 82% | 83% | 81% | 85% | 76% | 81% |
| bitwise_and            | 38% | 37% | 0% | 38% | 42% | 86% | 41% | 38% | 0% |
| bitwise_not            | 45% | 49% | 0% | 45% | 45% | 0% | 50% | 46% | 0% |
| bitwise_or             | 42% | 39% | 0% | 41% | 42% | 91% | 49% | 45% | 0% |
| bitwise_xor            | 41% | 41% | 0% | 41% | 41% | 91% | 44% | 99% | 0% |
| cholesky               | 48% | 47% | 0% | 47% | 48% | 49% | 34% | 34% | 0% |
| cross                  | 43% | 86% | 86% | 43% | 43% | 43% | 45% | 45% | 86% |
| diag_part              | 43% | 42% | 0% | 50% | 43% | 0% | 50% | 50% | 0% |
| div                    | 34% | 34% | 0% | 32% | 33% | 34% | 35% | 64% | 32% |
| div_no_nan             | 47% | 47% | 0% | 0% | 47% | 0% | 0% | 47% | 0% |
| dot                    | 96% | 99% | 50% | 96% | 96% | 96% | 99% | 95% | 96% |
| equal                  | 84% | 79% | 79% | 84% | 84% | 84% | 84% | 94% | 79% |
| exp                    | 45% | 89% | 0% | 45% | 45% | 38% | 49% | 38% | 0% |
| expm1                  | 43% | 47% | 0% | 42% | 43% | 43% | 44% | 79% | 34% |
| eye                    | 94% | 46% | 39% | 94% | 94% | 94% | 50% | 79% | 39% |
| floor_div              | 34% | 34% | 31% | 34% | 33% | 0% | 33% | 32% | 31% |
| floor_mod              | 35% | 35% | 0% | 35% | 35% | 34% | 35% | 35% | 22% |
| ger                    | 46% | 98% | 0% | 46% | 44% | 96% | 0% | 44% | 0% |
| greater                | 84% | 87% | 66% | 84% | 86% | 83% | 86% | 94% | 66% |
| greater_equal          | 85% | 85% | 68% | 83% | 85% | 68% | 95% | 97% | 68% |
| less                   | 77% | 77% | 50% | 50% | 77% | 76% | 77% | 86% | 50% |
| less_equal             | 84% | 84% | 50% | 84% | 85% | 84% | 85% | 94% | 78% |
| log                    | 92% | 95% | 94% | 92% | 95% | 94% | 100% | 100% | 91% |
| log1p                  | 95% | 95% | 95% | 95% | 95% | 95% | 100% | 93% | 90% |
| logical_and            | 41% | 45% | 0% | 41% | 41% | 41% | 87% | 50% | 41% |
| logical_not            | 95% | 50% | 94% | 50% | 50% | 95% | 50% | 50% | 94% |
| logical_or             | 78% | 90% | 89% | 91% | 91% | 89% | 100% | 96% | 89% |
| mod                    | 34% | 35% | 34% | 35% | 35% | 35% | 34% | 0% | 34% |
| mul                    | 83% | 88% | 60% | 81% | 83% | 78% | 85% | 41% | 60% |
| neg                    | 45% | 50% | 44% | 45% | 45% | 44% | 50% | 50% | 44% |
| not_equal              | 75% | 92% | 72% | 75% | 87% | 85% | 87% | 97% | 72% |
| pow                    | 39% | 39% | 6% | 39% | 31% | 39% | 40% | 37% | 37% |
| pows                   | 86% | 86% | 33% | 86% | 86% | 45% | 86% | 95% | 33% |
| real_div               | 82% | 82% | 50% | 79% | 50% | 50% | 100% | 100% | 50% |
| reciprocal             | 46% | 46% | 86% | 46% | 44% | 100% | 50% | 96% | 86% |
| rsqrt                  | 96% | 96% | 86% | 44% | 46% | 94% | 96% | 50% | 86% |
| rsqrt_grad             | 44% | 44% | 0% | 50% | 44% | 0% | 44% | 50% | 0% |
| sign                   | 96% | 95% | 95% | 96% | 96% | 98% | 96% | 98% | 95% |
| sqrt                   | 45% | 45% | 38% | 45% | 45% | 0% | 46% | 45% | 38% |
| sqrt_grad              | 0% | 43% | 0% | 0% | 27% | 0% | 0% | 0% | 0% |
| square                 | 41% | 49% | 41% | 46% | 44% | 41% | 49% | 41% | 41% |
| squared_difference     | 46% | 46% | 0% | 42% | 0% | 0% | 0% | 50% | 0% |
| sub                    | 82% | 92% | 75% | 81% | 82% | 82% | 91% | 75% | 75% |
| trace                  | 46% | 46% | 0% | 46% | 46% | 0% | 48% | 46% | 0% |

---

## 4. 胜负分析 (ATTest vs B3)

| 模型 | ATTest 胜 | B3 胜 | 持平 | ATTest 平均 overall | B3 平均 overall |
|------|----------:|------:|-----:|--------------------:|----------------:|
| qwen  |      2 |    39 |     4 |                38.6% |             63.6% |
| ds    |     12 |    20 |    13 |                56.1% |             57.4% |
| glm   |      7 |    34 |     4 |                42.7% |             62.6% |

---

## 5. 关键发现

### 5.1 ATTest vs B3 覆盖率差距

- **qwen**: ATTest=38.6% vs B3=63.6% (delta=-25.0%)
- **ds**: ATTest=56.1% vs B3=57.4% (delta=-1.3%)
- **glm**: ATTest=42.7% vs B3=62.6% (delta=-19.9%)

### 5.2 ATTest 零覆盖率算子分析

- **qwen** (17 ops): bitwise_and, bitwise_not, bitwise_or, bitwise_xor, cholesky, diag_part, div, div_no_nan, exp, expm1, floor_mod, ger, logical_and, rsqrt_grad, sqrt_grad, squared_difference, trace
- **ds** (9 ops): bitwise_not, diag_part, div_no_nan, floor_div, rsqrt_grad, sqrt, sqrt_grad, squared_difference, trace
- **glm** (14 ops): abs, bitwise_and, bitwise_not, bitwise_or, bitwise_xor, cholesky, diag_part, div_no_nan, exp, ger, rsqrt_grad, sqrt_grad, squared_difference, trace

### 5.3 ATTest 优势算子 (overall > B3 + 5%)

- **qwen** (2 ops): logical_not(+44%), reciprocal(+40%)
- **ds** (8 ops): bitwise_and(+45%), bitwise_or(+50%), bitwise_xor(+50%), ger(+52%), logical_not(+45%), pow(+8%), reciprocal(+56%), rsqrt(+49%)
- **glm** (6 ops): add(+5%), cross(+41%), logical_not(+44%), mod(+34%), mul(+19%), rsqrt(+36%)

### 5.4 耗时对比

| 模型 | B2 | B3 | ATTest | ATTest/B3 |
|------|---:|---:|-------:|----------:|
| qwen  | 28h | 27h | 12h | 12h/27h |
| ds    | 27h | 21h | 69h | 69h/21h |
| glm   | 61h | 54h | 112h | 112h/54h |

### 5.5 非零算子子集对比 (排除已知 bug, 待修复)

以 ATTest overall > 0% 的算子为子集，B2/B3 取相同子集计算均值 (overall = (host+api)/2, N/A 按 0 计)。

| 模型 | N | ATTest overall | B2 overall | B3 overall | delta vs B3 | 胜 (vs B3) | 负 | 持平 |
|------|--:|---------------:|-----------:|-----------:|------------:|-----------:|---:|-----:|
| qwen  | 28 |         62.0% |      73.6% |      72.5% |      -10.5% |          2 | 22 |    4 |
| ds    | 36 |         70.1% |      62.2% |      62.6% |       +7.6% |         12 | 12 |   12 |
| glm   | 31 |         61.9% |      70.9% |      70.4% |      -8.5% |          7 | 21 |    3 |

- **DS 模型**: 排除 0% bug 后 ATTest (70.1%) 反超 B3 (+7.6%)，胜负持平，host 层覆盖率优势充分体现
- **Qwen/GLM**: 差距从 20-25% 缩小到 8-10%；仍有部分算子 ATTest api 层偏低 (如 dot, less, less_equal, pows)

---

## 6. 结论

1. **全量 45 算子下**，ATTest 在三模型上低于统一口径的 B2/B3 基线 (qwen -25%, glm -20%, ds -1.3%)
2. **排除零覆盖率 bug 后**，ATTest 在 ds 上反超 B3 (+7.6%)，qwen/glm 差距缩小到 8-10%
3. 零覆盖率 (qwen 17, glm 14, ds 9 ops) 根因: 头文件错误、CMake 配置错误、测试运行时崩溃 (已知 bug, 待修复)
4. **Rollback 机制已验证生效**: Qwen 16 ops、DS 14 ops、GLM 12 ops 触发了 epoch rollback
5. ATTest 在 host 层覆盖率优势明显 (ds 模型: bitwise_and/or/xor, ger, logical_not, reciprocal, rsqrt 均 +45% 以上)
6. 统计口径统一为 overall = (host+api)/2, N/A 按 0 计；B2/B3 因 host 缺失整体被拉低，ATTest 与基线差距收窄

---
