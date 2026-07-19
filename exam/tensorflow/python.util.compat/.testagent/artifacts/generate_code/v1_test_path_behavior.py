#!/usr/bin/env python3
"""Test Path object behavior."""

from pathlib import Path

# Test Path object behavior
path_obj = Path("./test")
print(f"Path('./test') = {path_obj}")
print(f"str(Path('./test')) = {str(path_obj)}")
print(f"Path('./test').__fspath__() = {path_obj.__fspath__()}")
print(f"Path('./test').as_posix() = {path_obj.as_posix()}")

# Test with absolute path
abs_path = Path("/tmp/test")
print(f"\nPath('/tmp/test') = {abs_path}")
print(f"str(Path('/tmp/test')) = {str(abs_path)}")
print(f"Path('/tmp/test').__fspath__() = {abs_path.__fspath__()}")

# Test with parent directory
parent_path = Path("../parent/test")
print(f"\nPath('../parent/test') = {parent_path}")
print(f"str(Path('../parent/test')) = {str(parent_path)}")
print(f"Path('../parent/test').__fspath__() = {parent_path.__fspath__()}")