import os

# 删除临时测试文件
temp_files = ['test_value_range_behavior.py', 'run_test.py', 'cleanup.py', 'test_integer_dtype.py']
for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted {file}")

print("Cleanup completed.")