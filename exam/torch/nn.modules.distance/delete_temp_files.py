import os

files_to_delete = ["test_dim_behavior.py", "run_test.py", "cleanup.py", "delete_temp_files.py"]
for f in files_to_delete:
    try:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted: {f}")
    except Exception as e:
        print(f"Error deleting {f}: {e}")