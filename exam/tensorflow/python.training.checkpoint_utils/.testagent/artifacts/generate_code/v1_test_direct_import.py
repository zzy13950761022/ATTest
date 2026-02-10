#!/usr/bin/env python3
"""Test direct import of tensorflow.python."""

import sys

try:
    import tensorflow.python
    print("✓ Successfully imported tensorflow.python")
    print(f"  tensorflow.python module: {tensorflow.python}")
except Exception as e:
    print(f"✗ Failed to import tensorflow.python: {e}")
    import traceback
    traceback.print_exc()

try:
    import tensorflow.python.training
    print("✓ Successfully imported tensorflow.python.training")
except Exception as e:
    print(f"✗ Failed to import tensorflow.python.training: {e}")
    import traceback
    traceback.print_exc()

try:
    import tensorflow.python.training.checkpoint_management
    print("✓ Successfully imported tensorflow.python.training.checkpoint_management")
except Exception as e:
    print(f"✗ Failed to import tensorflow.python.training.checkpoint_management: {e}")
    import traceback
    traceback.print_exc()

try:
    import tensorflow.python.training.py_checkpoint_reader
    print("✓ Successfully imported tensorflow.python.training.py_checkpoint_reader")
except Exception as e:
    print(f"✗ Failed to import tensorflow.python.training.py_checkpoint_reader: {e}")
    import traceback
    traceback.print_exc()