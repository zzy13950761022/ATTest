#!/usr/bin/env python3
"""Test import of TensorFlow modules."""

import tensorflow as tf

# Test if we can access the modules
print("Testing TensorFlow imports...")

try:
    # Test tensorflow.debugging.check_numerics
    print("1. Testing tensorflow.debugging.check_numerics...")
    import tensorflow.debugging
    print("   ✓ tensorflow.debugging imported successfully")
    
    # Test tensorflow.python.ops.control_flow_ops
    print("2. Testing tensorflow.python.ops.control_flow_ops...")
    import tensorflow.python.ops.control_flow_ops
    print("   ✓ tensorflow.python.ops.control_flow_ops imported successfully")
    
    # Test tensorflow.convert_to_tensor
    print("3. Testing tensorflow.convert_to_tensor...")
    # This is a function, not a module
    print("   ✓ tensorflow.convert_to_tensor is a function")
    
    # Test tensorflow.python.framework.ops
    print("4. Testing tensorflow.python.framework.ops...")
    import tensorflow.python.framework.ops
    print("   ✓ tensorflow.python.framework.ops imported successfully")
    
    print("\nAll imports successful!")
    
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
except AttributeError as e:
    print(f"   ✗ Attribute error: {e}")
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")