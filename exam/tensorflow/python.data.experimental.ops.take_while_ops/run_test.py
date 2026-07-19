import subprocess
import sys

# Run the test file
result = subprocess.run([sys.executable, "test_actual_behavior.py"], 
                       capture_output=True, text=True)
print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")