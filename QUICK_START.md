# 🚀 ATTest Batch Testing Quick Start

## ✅ Environment Ready

- ✓ Python 3.10.19
- ✓ PyTorch 1.13.0
- ✓ TensorFlow 2.9.0
- ✓ ATTest 0.1.0

See `ENVIRONMENT_VERIFICATION_REPORT.md` for details.

---

## 🎯 Start Batch Testing PyTorch Modules

### Method 1: Interactive Launcher (Simplest)

```bash
./run_batch_test.sh
```

**Choose an action:**
- `1` → Start or continue testing (5 epochs)
- `2` → Start or continue testing (3 epochs)
- `3` → Restart from scratch
- `6` → Test a single module

### Method 2: Command Line

```bash
# Start batch testing (default: 5 epochs)
/opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py

# Short version if PATH or an alias is already configured
python batch_test_torch.py
```

---

## 📊 Test Scope

### PyTorch Modules: **51**

Defined in `artifact/rundefinitions/pynguinml-torch.xml`

Example modules:
```
1. torch._linalg_utils
2. torch._lobpcg
3. torch._lowrank
4. torch._tensor_str
5. torch.ao.nn.quantized.functional
...
51. torch.utils.data.dataset
```

Run the full list inspection:
```bash
/opt/anaconda3/envs/attest-experiment/bin/python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('artifact/rundefinitions/pynguinml-torch.xml')
for i, m in enumerate(tree.findall('.//module'), 1):
    print(f'{i:2d}. {m.text}')
"
```

---

## 📁 Output Layout

```
exam/torch/
├── batch_test_state.json      # Testing progress
├── batch_test.log             # Detailed log
├── batch_test_report.md       # Test report
└── torch/
    ├── _linalg_utils/         # Module 1
    │   ├── .attest/
    │   │   ├── artifacts/
    │   │   └── state.json
    │   ├── tests/
    │   │   └── test_*.py
    │   ├── coverage.xml
    │   └── final_report.md
    ├── _lobpcg/               # Module 2
    └── ...                    # The other 49 modules
```

---

## 🔍 Monitor Progress

### Watch Logs in Real Time

```bash
# View execution log
tail -f exam/torch/batch_test.log

# View current state
cat exam/torch/batch_test_state.json | python -m json.tool
```

### View Progress Statistics

```bash
# Using jq, if installed
jq '.completed | length' exam/torch/batch_test_state.json
jq '.failed | length' exam/torch/batch_test_state.json

# Or using Python
python -c "
import json
with open('exam/torch/batch_test_state.json') as f:
    state = json.load(f)
    print(f'Completed: {len(state[\"completed\"])}')
    print(f'Failed: {len(state[\"failed\"])}')
    print(f'Current index: {state[\"current_index\"]}')
"
```

---

## ⏸️ Interrupt and Resume

### Interrupt a Run

Press `Ctrl+C` to stop the run. State is saved automatically.

### Resume a Run

```bash
# Re-run the same command and continue automatically
./run_batch_test.sh

# Or
/opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py
```

### Start from a Specific Position

```bash
# Start from the 10th module (0-based index)
/opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py --start 9

# Start from the 25th module
/opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py --start 24
```

### Restart from Scratch

```bash
# Clear progress and restart
/opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py --reset
```

---

## 🧪 Test a Single Module

### Method 1: Use the Launcher

```bash
./run_batch_test.sh
# Choose option 6: test a single module
# Input: torch.mean
```

### Method 2: Run the Command Directly

```bash
/opt/anaconda3/envs/attest-experiment/bin/python -m attest_cli.cli run \
  -f torch.mean \
  --workspace ./exam/torch/torch/mean \
  --mode full-auto \
  --epoch 5
```

---

## 📝 View Results

### Test Reports

```bash
# View the overall batch test report
cat exam/torch/batch_test_report.md

# View a single module report
cat exam/torch/torch/_linalg_utils/final_report.md
```

### Coverage Data

```bash
# View coverage for a single module
cat exam/torch/torch/_linalg_utils/coverage.xml
```

### Analysis Report

```bash
# View analysis output
cat exam/torch/torch/_linalg_utils/analysis.md
```

---

## ⚙️ Advanced Options

### Customize Epochs

```bash
# Use 3 epochs for a faster run
/opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py --epoch 3

# Use 10 epochs for a more thorough run
/opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py --epoch 10
```

### Specify a Workspace

```bash
# Use a different working directory
/opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py \
  --workspace ./my_custom_workspace
```

### Run in the Background

```bash
# Use nohup
nohup /opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py > batch.out 2>&1 &

# View output
tail -f batch.out
```

---

## ⏱️ Time Estimate

Typical durations:

| Item | Time |
|------|------|
| Single simple module | 3-5 minutes |
| Single complex module | 5-15 minutes |
| All 51 modules | **4-12 hours** |

**Recommendation**: Run in the background or inside a `tmux`/`screen` session.

---

## 🛠️ Troubleshooting

### Issue 1: `conda activate` Fails

**Solution**: Use the full interpreter path directly without `activate`.

See `CONDA_ACTIVATION.md` for details.

### Issue 2: Module Import Fails

```bash
# Verify the environment
/opt/anaconda3/envs/attest-experiment/bin/python -c "
import torch
import tensorflow as tf
print('OK')
"
```

### Issue 3: Insufficient Disk Space

```bash
# Check available space
df -h .

# Clean old versioned directories
find exam/torch -name "v[0-9]*" -type d -exec rm -rf {} +
```

### Issue 4: Test Timeout

Edit `batch_test_torch.py` and update the timeout value:

```python
timeout=3600  # Increase this value, for example to 7200 (2 hours)
```

---

## 📚 Related Documents

| Document | Description |
|------|------|
| `BATCH_TEST_GUIDE.md` | Detailed usage guide |
| `ENVIRONMENT_VERIFICATION_REPORT.md` | Environment verification report |
| `CONDA_ACTIVATION.md` | Conda activation notes |
| `MODIFICATION_SUMMARY.md` | Summary of code changes |

---

## 📞 Get Help

```bash
# Show batch testing help
/opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py --help

# Show launcher options
./run_batch_test.sh
```

---

## ✨ Quick Command Reference

```bash
# 1. Start testing
./run_batch_test.sh

# 2. Check progress
cat exam/torch/batch_test_state.json

# 3. View logs
tail -f exam/torch/batch_test.log

# 4. View reports
cat exam/torch/batch_test_report.md

# 5. Test a single module
./run_batch_test.sh  # Choose option 6

# 6. Verify the environment
./start_experiment_env.sh

# 7. Restart from scratch
/opt/anaconda3/envs/attest-experiment/bin/python batch_test_torch.py --reset
```

---

## 🎉 Start Testing

Everything is ready. You can start batch testing now:

```bash
./run_batch_test.sh
```

Good luck with the run.
