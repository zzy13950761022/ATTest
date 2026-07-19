#!/usr/bin/env python3
"""Clean up temporary test files."""
import os

files_to_remove = ['test_fix.py', 'run_test.sh', 'cleanup.py']
for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed {file}")

print("Cleanup completed!")