"""Test cases for tensorflow.python.ops.ctc_ops module."""

import numpy as np
import tensorflow as tf
import pytest

from tensorflow.python.ops import ctc_ops

# ==== BLOCK:HEADER START ====
"""Test cases for tensorflow.python.ops.ctc_ops module."""

import numpy as np
import tensorflow as tf
import pytest

from tensorflow.python.ops import ctc_ops


def create_sparse_labels(batch_size, max_time, num_labels, seed=42):
    """Create random sparse labels for testing with proper constraints."""
    np.random.seed(seed)
    indices = []
    values = []
    
    for b in range(batch_size):
        # Create short sequences to avoid "Not enough time for target transition sequence" error
        # CTC requires: 2 * seq_len + 1 <= sequence_length for valid paths
        # We'll use very short sequences (1-3 labels) for safety
        max_seq_len = min(3, max_time // 2)  # Ensure sequence fits
        if max_seq_len < 1:
            max_seq_len = 1
        
        seq_len = np.random.randint(1, max_seq_len + 1)
        # Random time positions within valid range
        times = np.sort(np.random.choice(max_time, seq_len, replace=False))
        # Random label values within valid range
        labels = np.random.randint(0, num_labels, seq_len)
        
        for t, label in zip(times, labels):
            indices.append([b, t])
            values.append(label)
    
    indices = np.array(indices, dtype=np.int64)
    values = np.array(values, dtype=np.int32)
    dense_shape = [batch_size, max_time]
    
    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape
    )


def create_simple_sparse_labels(batch_size, max_time, num_labels, seed=42):
    """Create very simple sparse labels for basic testing."""
    np.random.seed(seed)
    indices = []
    values = []
    
    for b in range(batch_size):
        # Always create exactly 1 label at a random time position
        t = np.random.randint(0, max_time)
        label = np.random.randint(0, num_labels)
        
        indices.append([b, t])
        values.append(label)
    
    indices = np.array(indices, dtype=np.int64)
    values = np.array(values, dtype=np.int32)
    
    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[batch_size, max_time]
    )


def create_logits(batch_size, max_time, num_classes, time_major=True, seed=42):
    """Create random logits for testing with moderate values."""
    np.random.seed(seed)
    
    # Use moderate values to avoid extreme logits that cause numerical issues
    scale = 0.1  # Scale down to avoid extreme values
    
    if time_major:
        # Shape: [max_time, batch_size, num_classes]
        logits = np.random.randn(max_time, batch_size, num_classes).astype(np.float32) * scale
    else:
        # Shape: [batch_size, max_time, num_classes]
        logits = np.random.randn(batch_size, max_time, num_classes).astype(np.float32) * scale
    
    return tf.constant(logits)


def create_sequence_length(batch_size, max_time, seed=42):
    """Create random sequence lengths for testing."""
    np.random.seed(seed)
    # Use full sequence length or near-full for reliable testing
    # This ensures labels fit within sequence_length
    lengths = np.full(batch_size, max_time, dtype=np.int32)
    return tf.constant(lengths, dtype=tf.int32)


def assert_loss_shape_and_properties(loss, batch_size):
    """Assert basic properties of CTC loss output."""
    # Check shape
    assert loss.shape == (batch_size,), f"Expected shape ({batch_size},), got {loss.shape}"
    
    # Check dtype
    assert loss.dtype == tf.float32, f"Expected float32, got {loss.dtype}"
    
    # Check finite values (no NaN or Inf)
    loss_np = loss.numpy()
    assert np.all(np.isfinite(loss_np)), f"Loss contains NaN or Inf values: {loss_np}"
    
    # Check non-negative (CTC loss should be non-negative)
    assert np.all(loss_np >= -1e-6), f"Loss contains negative values: {loss_np}"
    
    # Additional check: loss should not be extremely large for moderate inputs
    assert np.all(loss_np < 100.0), f"Loss values too large: {loss_np}"


def assert_decoder_outputs(decoded, log_probabilities, batch_size, top_paths):
    """Assert basic properties of beam search decoder output."""
    # Check decoded is a list
    assert isinstance(decoded, list), f"Expected list, got {type(decoded)}"
    assert len(decoded) == top_paths, f"Expected {top_paths} paths, got {len(decoded)}"
    
    # Check each decoded path is a SparseTensor
    for i, sparse_tensor in enumerate(decoded):
        assert isinstance(sparse_tensor, tf.SparseTensor), \
            f"Path {i}: Expected SparseTensor, got {type(sparse_tensor)}"
        
        # Check sparse tensor properties
        assert sparse_tensor.indices.shape[1] == 2, \
            f"Path {i}: Expected indices shape (N, 2), got {sparse_tensor.indices.shape}"
        assert sparse_tensor.dense_shape.shape == (2,), \
            f"Path {i}: Expected dense_shape shape (2,), got {sparse_tensor.dense_shape.shape}"
        assert sparse_tensor.dense_shape[0] == batch_size, \
            f"Path {i}: Expected batch_size {batch_size}, got {sparse_tensor.dense_shape[0]}"
    
    # Check log probabilities shape
    assert log_probabilities.shape == (batch_size, top_paths), \
        f"Expected log_probabilities shape ({batch_size}, {top_paths}), got {log_probabilities.shape}"
    
    # Check log probabilities dtype
    assert log_probabilities.dtype == tf.float32, \
        f"Expected float32, got {log_probabilities.dtype}"
    
    # Check log probabilities are finite
    log_probs_np = log_probabilities.numpy()
    assert np.all(np.isfinite(log_probs_np)), f"Log probabilities contain NaN or Inf values: {log_probs_np}"
# ==== BLOCK:HEADER END ====