import subprocess
import sys

# 运行测试并捕获输出
cmd = [sys.executable, "-m", "pytest", "tests/test_tensorflow_python_ops_histogram_ops.py", "-v"]
print(f"Running command: {' '.join(cmd)}")
print("=" * 80)

result = subprocess.run(cmd, capture_output=True, text=True)

# 打印输出
if result.stdout:
    print("STDOUT:")
    # 只显示测试结果部分
    lines = result.stdout.split('\n')
    for line in lines:
        if any(keyword in line for keyword in ['PASSED', 'FAILED', 'ERROR', 'test_', '====', 'passed', 'failed']):
            print(line)
    
print("\n" + "=" * 80)
print(f"Return code: {result.returncode}")

if result.returncode == 0:
    print("✅ 所有测试通过！")
else:
    print("❌ 有测试失败！")
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr[:500])