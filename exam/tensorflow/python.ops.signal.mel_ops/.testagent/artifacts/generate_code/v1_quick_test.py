#!/usr/bin/env python3
"""快速测试修复"""

import subprocess
import sys

# 运行修复验证脚本
print("运行修复验证脚本...")
result = subprocess.run([sys.executable, "test_fix_verification.py"], 
                       capture_output=True, text=True)
print("标准输出:")
print(result.stdout)
if result.stderr:
    print("标准错误:")
    print(result.stderr)
print(f"返回码: {result.returncode}")

if result.returncode == 0:
    print("\n✅ 修复验证成功！")
else:
    print("\n❌ 修复验证失败！")