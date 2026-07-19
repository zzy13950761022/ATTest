import os

# List of temporary files to clean up
temp_files = [
    "test_num_features.py",
    "run_test.py",
    "direct_test.py", 
    "execute_test.py",
    "cleanup.py",
    "cleanup_temp.py"
]

for file in temp_files:
    if os.path.exists(file):
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Failed to remove {file}: {e}")
    else:
        print(f"Not found: {file}")