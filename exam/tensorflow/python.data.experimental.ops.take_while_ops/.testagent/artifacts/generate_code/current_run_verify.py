import subprocess
import sys

# Run the verification script
print("Running verification script...")
result = subprocess.run([sys.executable, "verify_fix.py"], 
                       capture_output=True, text=True)
print("Output:")
print(result.stdout)
if result.stderr:
    print("Errors:")
    print(result.stderr)
print(f"Return code: {result.returncode}")