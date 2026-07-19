import subprocess
import sys

# Run pytest on the specific test file
result = subprocess.run(
    [sys.executable, "-m", "pytest", 
     "tests/test_tensorflow_python_ops_candidate_sampling_ops.py",
     "-v"],
    capture_output=True,
    text=True
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")