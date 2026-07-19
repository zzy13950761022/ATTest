import os

# Clean up temporary test files
files_to_remove = ['test_fixes.py', 'test_window_behavior.py', 'cleanup.py']
for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed {file}")

print("Cleanup complete.")