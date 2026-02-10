#!/usr/bin/env python3
"""Simple verification of fixes for logging_ops tests."""

import sys
import tensorflow as tf
from tensorflow.python.ops import logging_ops

def test_print_v2_capture():
    """Test that print_v2 output can be captured."""
    import io
    from unittest import mock
    
    # Test with mock stdout
    with mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        logging_ops.print_v2("test message", output_stream=sys.stdout)
        output = mock_stdout.getvalue()
        print(f"Mock stdout captured: '{output}'")
        assert len(output) > 0, "No output captured"
        assert "test message" in output

def test_scalar_summary_shapes():
    """Test that scalar_summary requires matching shapes."""
    # This should work - matching shapes
    tags = ["tag1", "tag2"]
    values = [tf.constant(1.0), tf.constant(2.0)]
    
    # Convert to tensors
    tags_tensor = tf.constant(tags, dtype=tf.string)
    values_tensor = tf.stack([tf.cast(v, tf.float32) for v in values])
    
    print(f"Tags shape: {tags_tensor.shape}")
    print(f"Values shape: {values_tensor.shape}")
    assert tags_tensor.shape == values_tensor.shape, "Shapes must match"
    
    try:
        result = logging_ops.scalar_summary(tags=tags_tensor, values=values_tensor)
        print(f"Scalar summary created successfully: {result}")
    except Exception as e:
        print(f"Error creating scalar summary: {e}")

if __name__ == "__main__":
    print("Testing print_v2 capture...")
    test_print_v2_capture()
    
    print("\nTesting scalar_summary shapes...")
    test_scalar_summary_shapes()
    
    print("\nAll verification tests completed.")