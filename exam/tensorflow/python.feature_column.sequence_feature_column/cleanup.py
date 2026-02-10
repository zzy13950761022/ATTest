import os

# Remove temporary files
temp_files = ["check_sequence_column.py", "cleanup.py"]
for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed {file}")