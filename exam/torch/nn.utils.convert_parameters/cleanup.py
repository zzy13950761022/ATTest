import os

# Remove temporary test files
temp_files = ["test_vector_length_mismatch.py", "run_test.py", "cleanup.py"]
for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed {file}")