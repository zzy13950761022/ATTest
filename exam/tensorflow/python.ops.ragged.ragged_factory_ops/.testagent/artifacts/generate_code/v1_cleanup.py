import os

files_to_remove = ['test_ragged_rank.py', 'run_test.py', 'verify_case04.py', 'cleanup.py']
for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed: {file}")
    else:
        print(f"Not found: {file}")