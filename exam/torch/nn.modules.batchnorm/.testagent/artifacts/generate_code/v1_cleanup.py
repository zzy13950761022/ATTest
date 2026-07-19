import os

files_to_remove = [
    "test_num_features.py",
    "run_test.py", 
    "direct_test.py",
    "execute_test.py",
    "cleanup.py"
]

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed: {file}")
    else:
        print(f"Not found: {file}")