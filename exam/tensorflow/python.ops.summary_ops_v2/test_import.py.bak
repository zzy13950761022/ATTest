#!/usr/bin/env python3
"""Test script to verify imports and basic functionality."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
    
    from tensorflow.python.ops import summary_ops_v2
    print("✓ summary_ops_v2 imported successfully")
    
    # Test basic functionality
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Eager execution: {tf.executing_eagerly()}")
    
    # Test that we can access the module
    print(f"\nModule members: {[m for m in dir(summary_ops_v2) if not m.startswith('_')][:10]}...")
    
    # Check if write function exists
    if hasattr(summary_ops_v2, 'write'):
        print("✓ write function found in summary_ops_v2")
    else:
        print("✗ write function NOT found in summary_ops_v2")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Other error: {e}")
    sys.exit(1)

print("\n✓ All imports successful!")