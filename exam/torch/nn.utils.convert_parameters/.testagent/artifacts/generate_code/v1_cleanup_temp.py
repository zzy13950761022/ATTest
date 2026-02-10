import os

# Remove temporary test files
temp_files = [
    "test_vector_length_behavior.py",
    "quick_test.py", 
    "direct_run.py",
    "run_test2.py",
    "cleanup_temp.py"
]

for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed {file}")