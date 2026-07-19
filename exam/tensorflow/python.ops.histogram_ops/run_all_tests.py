import subprocess
import sys

# 运行所有测试
result = subprocess.run([
    sys.executable, "-m", "pytest",
    "tests/test_tensorflow_python_ops_histogram_ops.py",
    "-v",
    "--tb=short",
    # "--cov=tests.test_tensorflow_python_ops_histogram_ops",
    # "--cov-report=term-missing"
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

# 检查测试结果
if result.returncode == 0:
    print("\n✅ 所有测试通过！")
else:
    print("\n❌ 有测试失败！")