#!/usr/bin/env python3
"""
ATTest batch testing script for PyTorch modules
Read all modules from the PynguinML XML definition file and run tests automatically
"""

import sys
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import json
import time
from typing import List, Dict, Optional

class BatchTester:
    def __init__(
        self,
        xml_file: str,
        workspace_base: str,
        python_path: str,
        epochs: int = 5,
        mode: str = "full-auto"
    ):
        self.xml_file = Path(xml_file)
        self.workspace_base = Path(workspace_base)
        self.python_path = python_path
        self.epochs = epochs
        self.mode = mode

        # State file
        self.state_file = self.workspace_base / "batch_test_state.json"
        self.log_file = self.workspace_base / "batch_test.log"

        # Ensure the workspace exists
        self.workspace_base.mkdir(parents=True, exist_ok=True)

        # Load state
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """Load testing state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "completed": [],
            "failed": [],
            "current_index": 0,
            "start_time": None,
            "last_update": None
        }

    def save_state(self):
        """Save testing state"""
        self.state["last_update"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def log(self, message: str, level: str = "INFO"):
        """Record logs"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)

        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def extract_modules(self) -> List[str]:
        """Extract the module list from the XML file"""
        try:
            tree = ET.parse(self.xml_file)
            root = tree.getroot()

            modules = []
            for module in root.findall('.//module'):
                if module.text:
                    modules.append(module.text.strip())

            return modules
        except Exception as e:
            self.log(f"Failed to parse the XML file: {e}", "ERROR")
            return []

    def module_to_workspace(self, module: str) -> Path:
        """Convert a module name to a workspace path."""
        # Replace dots with slashes, for example `torch.nn.modules.batchnorm` -> `torch/nn.modules.batchnorm`
        workspace_name = module.replace(".", "/")
        return self.workspace_base / workspace_name

    def run_test(self, module: str) -> bool:
        """Run tests for a single module"""
        workspace = self.module_to_workspace(module)

        self.log(f"Start testing module: {module}")
        self.log(f"Workspace: {workspace}")

        # Build command
        cmd = [
            self.python_path,
            "-m", "attest_cli.cli",
            "run",
            "--func", module,
            "--workspace", str(workspace),
            "--mode", self.mode,
            "--epoch", str(self.epochs)
        ]

        self.log(f"Run command: {' '.join(cmd)}")

        try:
            # Run tests
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1-hour timeout
            )
            elapsed = time.time() - start_time

            if result.returncode == 0:
                self.log(f"✓ Module {module} tests passed (elapsed: {elapsed:.1f}s)", "SUCCESS")
                return True
            else:
                self.log(f"✗ Module {module} tests failed (exit code: {result.returncode})", "ERROR")
                self.log(f"STDOUT: {result.stdout[-500:]}", "DEBUG")
                self.log(f"STDERR: {result.stderr[-500:]}", "DEBUG")
                return False

        except subprocess.TimeoutExpired:
            self.log(f"✗ Module {module} test timed out", "ERROR")
            return False
        except Exception as e:
            self.log(f"✗ Module {module} execution exception: {e}", "ERROR")
            return False

    def run_batch(self, start_index: Optional[int] = None):
        """Run tests in batch."""
        modules = self.extract_modules()

        if not modules:
            self.log("No modules were found", "ERROR")
            return

        total = len(modules)
        self.log(f"Found {total} modules to test")

        # Determine the starting position
        if start_index is not None:
            current = start_index
        else:
            current = self.state.get("current_index", 0)

        # Record the start time on the first run
        if not self.state.get("start_time"):
            self.state["start_time"] = datetime.now().isoformat()

        self.log(f"Start from module {current + 1}/{total}")

        # Test modules one by one
        for i in range(current, total):
            module = modules[i]

            # Check whether the module is already completed
            if module in self.state["completed"]:
                self.log(f"Skip completed module: {module}", "INFO")
                continue

            self.log(f"\n{'='*60}")
            self.log(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            self.log(f"Module: {module}")
            self.log(f"{'='*60}")

            # Run tests
            success = self.run_test(module)

            # Update state
            if success:
                self.state["completed"].append(module)
            else:
                self.state["failed"].append(module)

            self.state["current_index"] = i + 1
            self.save_state()

            # Short pause
            time.sleep(2)

        # Generate report
        self.generate_report(modules)

    def generate_report(self, modules: List[str]):
        """Generate the test report"""
        total = len(modules)
        completed = len(self.state["completed"])
        failed = len(self.state["failed"])
        remaining = total - completed - failed

        report_file = self.workspace_base / "batch_test_report.md"

        report = f"""# ATTest Batch Test Report - PyTorch Modules

## Test Overview

- **Start time**: {self.state.get('start_time', 'N/A')}
- **End time**: {datetime.now().isoformat()}
- **Total modules**: {total}
- **Completed**: {completed} ({completed/total*100:.1f}%)
- **Failed**: {failed} ({failed/total*100:.1f}%)
- **Not tested**: {remaining} ({remaining/total*100:.1f}%)

## Test Configuration

- **XML file**: {self.xml_file}
- **Workspace**: {self.workspace_base}
- **Mode**: {self.mode}
- **Iteration rounds**: {self.epochs}

## Successful Modules ({completed})

"""
        for module in self.state["completed"]:
            workspace = self.module_to_workspace(module)
            report += f"- ✓ `{module}` → `{workspace}`\n"

        if self.state["failed"]:
            report += f"\n## Failed Modules ({failed})\n\n"
            for module in self.state["failed"]:
                workspace = self.module_to_workspace(module)
                report += f"- ✗ `{module}` → `{workspace}`\n"

        if remaining > 0:
            report += f"\n## Not Tested Modules ({remaining})\n\n"
            untested = [m for m in modules if m not in self.state["completed"] and m not in self.state["failed"]]
            for module in untested:
                report += f"- ⊙ `{module}`\n"

        report += "\n## Detailed Log\n\n"
        report += f"See the full log: `{self.log_file}`\n"

        with open(report_file, 'w') as f:
            f.write(report)

        self.log(f"\n{'='*60}")
        self.log("Testing completed!")
        self.log(f"Report generated: {report_file}")
        self.log(f"{'='*60}\n")

        # Print summary
        print(report)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ATTest batch testing tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all PyTorch modules
  python batch_test_torch.py

  # Start from the 10th module
  python batch_test_torch.py --start 9

  # Specify a different epoch count
  python batch_test_torch.py --epoch 3

  # Use interactive mode
  python batch_test_torch.py --mode interactive
        """
    )

    parser.add_argument(
        "--xml",
        default="artifact/rundefinitions/pynguinml-torch.xml",
        help="Path to the XML definition file (default: `artifact/rundefinitions/pynguinml-torch.xml`)"
    )
    parser.add_argument(
        "--workspace",
        default="./exam/torch",
        help="Base workspace path (default: ./exam/torch)"
    )
    parser.add_argument(
        "--python",
        default="/opt/anaconda3/envs/attest-experiment/bin/python",
        help="Python interpreter path"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=5,
        help="Iteration rounds (default: 5)"
    )
    parser.add_argument(
        "--mode",
        choices=["full-auto", "interactive"],
        default="full-auto",
        help="Run mode (default: full-auto)"
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Start from the specified index (0-based)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset state and start from scratch"
    )

    args = parser.parse_args()

    # Create the tester
    tester = BatchTester(
        xml_file=args.xml,
        workspace_base=args.workspace,
        python_path=args.python,
        epochs=args.epoch,
        mode=args.mode
    )

    # Reset the state
    if args.reset:
        tester.log("Reset test state", "INFO")
        tester.state = {
            "completed": [],
            "failed": [],
            "current_index": 0,
            "start_time": None,
            "last_update": None
        }
        tester.save_state()

    # Run batch testing
    try:
        tester.run_batch(start_index=args.start)
    except KeyboardInterrupt:
        tester.log("\n\nTesting interrupted by the user", "WARNING")
        tester.save_state()
        tester.log("State has been saved; you can continue with the current configuration", "INFO")
        sys.exit(1)
    except Exception as e:
        tester.log(f"Batch testing exception: {e}", "ERROR")
        import traceback
        tester.log(traceback.format_exc(), "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
