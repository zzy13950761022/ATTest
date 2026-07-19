import subprocess
import sys

print("执行验证脚本...")
result = subprocess.run([sys.executable, "verify_fix.py"], 
                       capture_output=True, text=True, encoding='utf-8')
print("输出:")
print(result.stdout)
if result.stderr:
    print("错误:")
    print(result.stderr)
print("返回码:", result.returncode)