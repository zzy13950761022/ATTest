# 🚀 TestAgent 批量测试快速开始

## ✅ 环境已就绪

- ✓ Python 3.10.19
- ✓ PyTorch 1.13.0
- ✓ TensorFlow 2.9.0
- ✓ TestAgent 0.1.0

详见: `ENVIRONMENT_VERIFICATION_REPORT.md`

---

## 🎯 开始批量测试 PyTorch 模块

### 方法1: 交互式启动器（最简单）

```bash
./run_batch_test.sh
```

**选择操作:**
- 1 → 开始/继续测试（5 epochs）
- 2 → 开始/继续测试（3 epochs）
- 3 → 重新开始
- 6 → 测试单个模块

### 方法2: 命令行

```bash
# 开始批量测试（默认5 epochs）
/opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py

# 简化版本（如果已设置PATH或别名）
python batch_test_torch.py
```

---

## 📊 测试范围

### PyTorch模块: **51个**

来自 `artifact/rundefinitions/pynguinml-torch.xml`

示例模块:
```
1. torch._linalg_utils
2. torch._lobpcg
3. torch._lowrank
4. torch._tensor_str
5. torch.ao.nn.quantized.functional
...
51. torch.utils.data.dataset
```

完整列表运行:
```bash
/opt/anaconda3/envs/testagent-experiment/bin/python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('artifact/rundefinitions/pynguinml-torch.xml')
for i, m in enumerate(tree.findall('.//module'), 1):
    print(f'{i:2d}. {m.text}')
"
```

---

## 📁 输出结构

```
exam/torch/
├── batch_test_state.json      # 测试进度
├── batch_test.log             # 详细日志
├── batch_test_report.md       # 测试报告
└── torch/
    ├── _linalg_utils/         # 模块1
    │   ├── .testagent/
    │   │   ├── artifacts/
    │   │   └── state.json
    │   ├── tests/
    │   │   └── test_*.py
    │   ├── coverage.xml
    │   └── final_report.md
    ├── _lobpcg/               # 模块2
    └── ...                    # 其他49个模块
```

---

## 🔍 监控进度

### 实时查看日志

```bash
# 查看执行日志
tail -f exam/torch/batch_test.log

# 查看当前状态
cat exam/torch/batch_test_state.json | python -m json.tool
```

### 查看进度统计

```bash
# 使用jq（如果已安装）
jq '.completed | length' exam/torch/batch_test_state.json
jq '.failed | length' exam/torch/batch_test_state.json

# 或使用Python
python -c "
import json
with open('exam/torch/batch_test_state.json') as f:
    state = json.load(f)
    print(f'已完成: {len(state[\"completed\"])}')
    print(f'失败: {len(state[\"failed\"])}')
    print(f'当前索引: {state[\"current_index\"]}')
"
```

---

## ⏸️ 中断与恢复

### 中断测试

按 `Ctrl+C` 中断，状态会自动保存。

### 恢复测试

```bash
# 直接运行相同命令，自动从上次位置继续
./run_batch_test.sh

# 或
/opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py
```

### 从特定位置开始

```bash
# 从第10个模块开始（索引从0开始）
/opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py --start 9

# 从第25个模块开始
/opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py --start 24
```

### 重新开始

```bash
# 清除进度，从头开始
/opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py --reset
```

---

## 🧪 测试单个模块

### 方法1: 使用启动器

```bash
./run_batch_test.sh
# 选择 6: 测试单个模块
# 输入: torch.mean
```

### 方法2: 直接命令

```bash
/opt/anaconda3/envs/testagent-experiment/bin/python -m testagent_cli.cli run \
  -f torch.mean \
  --workspace ./exam/torch/torch/mean \
  --mode full-auto \
  --epoch 5
```

---

## 📝 查看结果

### 测试报告

```bash
# 查看批量测试总报告
cat exam/torch/batch_test_report.md

# 查看单个模块报告
cat exam/torch/torch/_linalg_utils/final_report.md
```

### 覆盖率数据

```bash
# 查看单个模块覆盖率
cat exam/torch/torch/_linalg_utils/coverage.xml
```

### 分析报告

```bash
# 查看分析结果
cat exam/torch/torch/_linalg_utils/analysis.md
```

---

## ⚙️ 高级选项

### 自定义epochs

```bash
# 使用3个epoch（更快）
/opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py --epoch 3

# 使用10个epoch（更彻底）
/opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py --epoch 10
```

### 指定工作目录

```bash
# 使用不同的工作目录
/opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py \
  --workspace ./my_custom_workspace
```

### 后台运行

```bash
# 使用nohup
nohup /opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py > batch.out 2>&1 &

# 查看输出
tail -f batch.out
```

---

## ⏱️ 时间估算

基于经验值：

| 项目 | 时间 |
|------|------|
| 单个简单模块 | 3-5分钟 |
| 单个复杂模块 | 5-15分钟 |
| 全部51个模块 | **4-12小时** |

**建议**: 使用后台运行或tmux/screen会话

---

## 🛠️ 故障排除

### 问题1: conda activate失败

**解决**: 使用完整路径，无需activate

详见: `CONDA_ACTIVATION.md`

### 问题2: 模块导入失败

```bash
# 验证环境
/opt/anaconda3/envs/testagent-experiment/bin/python -c "
import torch
import tensorflow as tf
print('OK')
"
```

### 问题3: 磁盘空间不足

```bash
# 检查空间
df -h .

# 清理旧版本文件
find exam/torch -name "v[0-9]*" -type d -exec rm -rf {} +
```

### 问题4: 测试超时

编辑 `batch_test_torch.py`，修改timeout值：
```python
timeout=3600  # 改为更大的值，如7200（2小时）
```

---

## 📚 相关文档

| 文档 | 说明 |
|------|------|
| `BATCH_TEST_GUIDE.md` | 详细使用指南 |
| `ENVIRONMENT_VERIFICATION_REPORT.md` | 环境验证报告 |
| `CONDA_ACTIVATION.md` | Conda激活说明 |
| `MODIFICATION_SUMMARY.md` | 代码修改总结 |

---

## 📞 获取帮助

```bash
# 查看批量测试帮助
/opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py --help

# 查看启动器选项
./run_batch_test.sh
```

---

## ✨ 快速命令参考

```bash
# 1. 开始测试
./run_batch_test.sh

# 2. 查看进度
cat exam/torch/batch_test_state.json

# 3. 查看日志
tail -f exam/torch/batch_test.log

# 4. 查看报告
cat exam/torch/batch_test_report.md

# 5. 测试单个模块
./run_batch_test.sh  # 选择选项6

# 6. 环境验证
./start_experiment_env.sh

# 7. 重新开始
/opt/anaconda3/envs/testagent-experiment/bin/python batch_test_torch.py --reset
```

---

## 🎉 开始测试吧！

一切就绪，现在可以开始批量测试了：

```bash
./run_batch_test.sh
```

祝测试顺利！ 🚀
