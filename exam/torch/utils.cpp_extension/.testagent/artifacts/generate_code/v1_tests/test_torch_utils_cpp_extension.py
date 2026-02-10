"""
主测试文件 - 导入并运行所有分组测试
"""
import pytest
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==== BLOCK:HEADER START ====
# 导入分组测试模块
try:
    from test_torch_utils_cpp_extension_g1 import *
    G1_AVAILABLE = True
except ImportError:
    G1_AVAILABLE = False
    print("Warning: G1 test module not available")

try:
    from test_torch_utils_cpp_extension_g2 import *
    G2_AVAILABLE = True
except ImportError:
    G2_AVAILABLE = False
    print("Warning: G2 test module not available")

# 运行所有测试的入口函数
def run_all_tests():
    """运行所有测试"""
    test_files = []
    
    if G1_AVAILABLE:
        test_files.append('test_torch_utils_cpp_extension_g1.py')
    
    if G2_AVAILABLE:
        test_files.append('test_torch_utils_cpp_extension_g2.py')
    
    if not test_files:
        print("No test files found!")
        return 1
    
    # 运行pytest
    return pytest.main([file for file in test_files] + ['-v'])
# ==== BLOCK:HEADER END ====

# ==== BLOCK:FOOTER START ====
if __name__ == "__main__":
    sys.exit(run_all_tests())
# ==== BLOCK:FOOTER END ====