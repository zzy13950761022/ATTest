import ast
import sys

try:
    with open("tests/test_tensorflow_python_ops_histogram_ops.py", "r") as f:
        content = f.read()
    
    # 检查语法
    ast.parse(content)
    print("✅ 语法检查通过！")
    
except SyntaxError as e:
    print(f"❌ 语法错误: {e}")
    print(f"错误位置: 第{e.lineno}行, 第{e.offset}列")
    
except Exception as e:
    print(f"❌ 其他错误: {e}")