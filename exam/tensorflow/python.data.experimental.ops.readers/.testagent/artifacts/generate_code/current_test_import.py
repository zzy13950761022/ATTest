import sys
sys.path.insert(0, '.')
try:
    import tests.test_tensorflow_python_data_experimental_ops_readers_tfrecord as test_module
    print("导入成功!")
    print(f"模块名称: {test_module.__name__}")
    
    # 检查类是否存在
    if hasattr(test_module, 'TestTFRecordReaders'):
        print("TestTFRecordReaders类存在")
        
        # 检查测试方法是否存在
        test_class = test_module.TestTFRecordReaders
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        print(f"找到测试方法: {test_methods}")
    else:
        print("TestTFRecordReaders类不存在")
        
except Exception as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()