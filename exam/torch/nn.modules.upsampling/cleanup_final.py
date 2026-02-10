import os
if os.path.exists("test_behavior.py"):
    os.remove("test_behavior.py")
if os.path.exists("cleanup_test.py"):
    os.remove("cleanup_test.py")
print("Cleaned up temporary files")