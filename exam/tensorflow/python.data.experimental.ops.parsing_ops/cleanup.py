"""Clean up temporary files"""
import os

files_to_remove = [
    "test_parse_example_dataset_usage.py",
    "run_test.py", 
    "fix_header.py",
    "verify_fix.py",
    "execute_verify.py",
    "correct_implementation.py",
    "cleanup.py"
]

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed: {file}")
    else:
        print(f"Not found: {file}")

print("Cleanup complete!")