import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorflow as tf
    print("TensorFlow imported successfully")
    
    # 尝试导入graph_io
    from tensorflow.python.framework import graph_io
    print("graph_io imported successfully")
    
    # 尝试导入file_io
    try:
        from tensorflow.python.lib.io import file_io
        print("file_io imported successfully")
    except ImportError as e:
        print(f"Failed to import file_io: {e}")
        
    # 检查tensorflow.python是否存在
    try:
        import tensorflow.python
        print("tensorflow.python exists")
    except ImportError as e:
        print(f"tensorflow.python does not exist: {e}")
        
except ImportError as e:
    print(f"Import error: {e}")