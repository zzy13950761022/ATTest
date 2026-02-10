import subprocess
import sys

# 运行特定的测试用例
result = subprocess.run([
    sys.executable, "-m", "pytest",
    "tests/test_tensorflow_python_ops_histogram_ops.py::test_histogram_fixed_width_bins_basic",
    "-v",
    "--tb=short"
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

# 运行 CASE_05 测试
print("\n" + "="*80 + "\n")
result2 = subprocess.run([
    sys.executable, "-m", "pytest",
    "tests/test_tensorflow_python_ops_histogram_ops.py::test_histogram_fixed_width_bins_invalid_value_range",
    "-v",
    "--tb=short"
], capture_output=True, text=True)

print("STDOUT:")
print(result2.stdout)
print("\nSTDERR:")
print(result2.stderr)
print(f"\nReturn code: {result2.returncode}")