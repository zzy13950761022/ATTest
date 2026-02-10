import subprocess
import sys

# Run the test
result = subprocess.run([sys.executable, "direct_test.py"], 
                       capture_output=True, text=True)

print("Output:")
print(result.stdout)
if result.stderr:
    print("Errors:")
    print(result.stderr)