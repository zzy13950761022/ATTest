import subprocess
import sys

# Run the specific tests that were failing
test_file = "tests/test_torch_nn_modules_upsampling_g3.py"
tests_to_run = [
    "test_parameter_exclusivity",
    "test_invalid_mode_parameter",
    "test_negative_scale_factor",
    "test_zero_scale_factor"
]

print("Running tests...")
for test_name in tests_to_run:
    print(f"\n=== Running {test_name} ===")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", f"{test_file}::{test_name}", "-v"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print(f"Return code: {result.returncode}")