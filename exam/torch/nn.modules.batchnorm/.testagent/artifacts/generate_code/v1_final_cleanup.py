import os
import sys

# Clean up all temporary files
files_to_remove = [
    "test_num_features.py",
    "run_test.py",
    "direct_test.py",
    "execute_test.py", 
    "cleanup.py",
    "cleanup_temp.py",
    "final_cleanup.py"
]

print("Cleaning up temporary files...")
for file in files_to_remove:
    if os.path.exists(file):
        try:
            os.remove(file)
            print(f"  ✓ Removed: {file}")
        except Exception as e:
            print(f"  ✗ Failed to remove {file}: {e}")
    else:
        print(f"  - Not found: {file}")

print("\nCleanup complete!")