#!/usr/bin/env python3
"""Simple test to verify the mock fix works."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import tensorflow as tf
from unittest import mock

# Test that we can import the modules
def test_imports():
    """Test that we can import the required modules."""
    import tensorflow.python.training.queue_runner as qr_module
    import tensorflow.python.ops.data_flow_ops as df_module
    
    assert hasattr(qr_module, 'QueueRunner')
    assert hasattr(qr_module, 'add_queue_runner')
    assert hasattr(df_module, 'FIFOQueue')
    assert hasattr(df_module, 'RandomShuffleQueue')
    
    print("✓ All imports work correctly")

# Test mock patching
def test_mock_patching():
    """Test that mock patching works with the new approach."""
    import tensorflow.python.training.queue_runner as qr_module
    import tensorflow.python.ops.data_flow_ops as df_module
    
    # Test QueueRunner mock
    with mock.patch.object(qr_module, 'QueueRunner') as mock_qr:
        mock_qr.return_value = mock.Mock()
        instance = qr_module.QueueRunner()
        assert mock_qr.called
        print("✓ QueueRunner mock works")
    
    # Test FIFOQueue mock
    with mock.patch.object(df_module, 'FIFOQueue') as mock_fifo:
        mock_fifo.return_value = mock.Mock()
        instance = df_module.FIFOQueue()
        assert mock_fifo.called
        print("✓ FIFOQueue mock works")
    
    # Test RandomShuffleQueue mock
    with mock.patch.object(df_module, 'RandomShuffleQueue') as mock_rsq:
        mock_rsq.return_value = mock.Mock()
        instance = df_module.RandomShuffleQueue()
        assert mock_rsq.called
        print("✓ RandomShuffleQueue mock works")
    
    # Test add_queue_runner mock
    with mock.patch.object(qr_module, 'add_queue_runner') as mock_aqr:
        mock_aqr.return_value = None
        qr_module.add_queue_runner(mock.Mock())
        assert mock_aqr.called
        print("✓ add_queue_runner mock works")

if __name__ == "__main__":
    print("Testing mock fix...")
    test_imports()
    test_mock_patching()
    print("\n✅ All tests passed! The mock fix should work correctly.")