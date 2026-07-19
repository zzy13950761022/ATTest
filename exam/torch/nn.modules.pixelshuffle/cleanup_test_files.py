import os

files_to_remove = [
    "test_actual_behavior.py",
    "test_pixelshuffle_errors.py",
    "run_test.py",
    "cleanup_test_files.py"
]

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed {file}")