#!/usr/bin/env python3
"""Clean up temporary files"""
import os

files_to_remove = [
    "test_validation.py",
    "run_validation.sh",
    "cleanup_temp.py"
]

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed: {file}")
    else:
        print(f"File not found: {file}")

print("Cleanup complete.")