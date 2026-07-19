import pytest
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 运行特定的测试
result = pytest.main([
    "tests/test_tensorflow_python_compiler_xla_xla_special.py::TestXLACompileSpecial::test_none_output_handling",
    "-v",
    "--tb=short"
])

print(f"\n测试结果: {result}")
print("如果返回0表示所有测试通过，非0表示有测试失败")