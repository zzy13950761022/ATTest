import subprocess
import sys

# 运行测试文件
result = subprocess.run([sys.executable, "test_parse_example_dataset_usage.py"], 
                       capture_output=True, text=True)
print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print("\nReturn code:", result.returncode)