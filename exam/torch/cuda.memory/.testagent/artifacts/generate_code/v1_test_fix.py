#!/usr/bin/env python3
"""测试修复后的代码"""

import pytest
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 运行高级测试
print("运行高级测试文件...")
result = pytest.main([
    "tests/test_torch_cuda_memory_advanced.py",
    "-v",
    "--tb=short"
])

print(f"\n测试结果: {result}")
print("如果返回0，表示所有测试通过")
print("如果返回1，表示有测试失败")
print("如果返回2，表示测试执行中断")
print("如果返回3，表示内部错误")
print("如果返回4，表示使用错误")
print("如果返回5，表示没有收集到测试")