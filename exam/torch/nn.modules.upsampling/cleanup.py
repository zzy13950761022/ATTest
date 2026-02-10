import os

# Clean up temporary files
files_to_remove = [
    "test_scale_factor_behavior.py",
    "run_test.py",
    "cleanup.py"
]

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed {file}")

print("Cleanup complete.")