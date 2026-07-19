import subprocess
import sys

# Run the vector length behavior test
print("Running vector length behavior test...")
result = subprocess.run([sys.executable, "test_vector_length_behavior.py"], 
                       capture_output=True, text=True)
print("Output:")
print(result.stdout)
if result.stderr:
    print("Errors:")
    print(result.stderr)