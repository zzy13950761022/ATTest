"""
Test cases for tensorflow.python.data.experimental.ops.grouping module.
Group G2: bucket_by_sequence_length sequence bucketing
"""

import pytest
import tensorflow as tf
import numpy as np
from tensorflow.python.data.experimental.ops.grouping import (
    bucket_by_sequence_length,
    group_by_reducer,
    group_by_window,
    Reducer
)
from tensorflow.python.framework import errors
import warnings

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G2

def create_variable_length_sequences(dataset_size=20, max_length=15):
    """Create dataset of variable length sequences."""
    sequences = []
    for i in range(dataset_size):
        # Random length between 1 and max_length
        length = np.random.randint(1, max_length + 1)
        # Create sequence with values
        sequence = list(range(i * 10, i * 10 + length))
        sequences.append(sequence)
    return tf.data.Dataset.from_generator(
        lambda: sequences,
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

def create_fixed_length_sequences(dataset_size=20, length=10):
    """Create dataset of fixed length sequences."""
    sequences = []
    for i in range(dataset_size):
        sequence = list(range(i * 10, i * 10 + length))
        sequences.append(sequence)
    return tf.data.Dataset.from_generator(
        lambda: sequences,
        output_signature=tf.TensorSpec(shape=(length,), dtype=tf.int32)
    )

def element_length_func(elem):
    """Default element length function - returns length of sequence."""
    return tf.shape(elem)[0]

def count_batches(dataset):
    """Count batches in a batched dataset."""
    count = 0
    for _ in dataset:
        count += 1
    return count

def get_batch_shapes(dataset):
    """Get shapes of all batches in dataset."""
    shapes = []
    for batch in dataset:
        if hasattr(batch, 'shape'):
            shapes.append(batch.shape)
        else:
            # For nested structures, get shape of first element
            shapes.append(tf.nest.map_structure(lambda x: x.shape, batch))
    return shapes

def dataset_to_list(dataset):
    """Convert dataset to Python list."""
    return list(dataset.as_numpy_iterator())

def verify_bucket_assignment(sequences, bucket_boundaries, element_length_func):
    """Verify that sequences are assigned to correct buckets."""
    bucket_assignments = []
    for seq in sequences:
        length = len(seq)
        bucket_idx = 0
        for boundary in bucket_boundaries:
            if length < boundary:
                break
            bucket_idx += 1
        bucket_assignments.append(bucket_idx)
    return bucket_assignments

# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: bucket_by_sequence_length 基本分桶
@pytest.mark.parametrize("bucket_boundaries, bucket_batch_sizes, padding_values, dataset_size", [
    ([5, 10, 15], [2, 2, 2, 2], None, 20),  # Basic case from test plan
])
def test_bucket_by_sequence_length_basic_bucketing(bucket_boundaries, bucket_batch_sizes, padding_values, dataset_size):
    """Test basic bucketing functionality of bucket_by_sequence_length."""
    
    # Create dataset of variable length sequences
    dataset = create_variable_length_sequences(dataset_size, max_length=20)
    
    # Weak assertion 1: returns_callable
    transform_fn = bucket_by_sequence_length(
        element_length_func=element_length_func,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        padding_values=padding_values
    )
    
    assert callable(transform_fn), "bucket_by_sequence_length should return a callable function"
    
    # Apply transformation to dataset
    transformed_dataset = transform_fn(dataset)
    
    # Weak assertion 2: dataset_interface
    assert hasattr(transformed_dataset, '__iter__'), "Transformed dataset should have iterator interface"
    assert hasattr(transformed_dataset, 'element_spec'), "Transformed dataset should have element_spec"
    
    # Get all batches
    batches = dataset_to_list(transformed_dataset)
    
    # Weak assertion 3: batch_sizes_correct
    # Check that batch sizes are correct (should be one of the bucket_batch_sizes)
    for batch in batches:
        batch_size = batch.shape[0]
        assert batch_size in bucket_batch_sizes, f"Batch size {batch_size} not in allowed sizes {bucket_batch_sizes}"
    
    # Weak assertion 4: sequences_bucketed
    # Verify that sequences with similar lengths are batched together
    # Count total number of sequences processed
    total_sequences = sum(batch.shape[0] for batch in batches)
    assert total_sequences <= dataset_size, f"Processed {total_sequences} sequences, expected at most {dataset_size}"
    
    # Additional verification: check that batches have consistent shapes within each batch
    for batch in batches:
        # All sequences in a batch should have the same length (after padding)
        assert batch.ndim == 2, f"Batch should be 2D, got shape {batch.shape}"
        # Check that padding is applied correctly (if any)
        if padding_values is not None:
            # If padding values specified, check they are used
            pass  # Would need to check actual padding values
    
    # Verify no data loss (all original sequences should appear in output)
    # This is a weak assertion - we just check we got some output
    assert len(batches) > 0, "Should get at least one batch"
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# TC-06: bucket_by_sequence_length 参数验证异常
@pytest.mark.parametrize("invalid_scenario, expected_exception", [
    ("mismatched_batch_sizes", ValueError),  # bucket_batch_sizes length mismatch
    ("non_increasing_boundaries", ValueError),  # Non-increasing bucket boundaries
])
def test_bucket_by_sequence_length_parameter_validation(invalid_scenario, expected_exception):
    """Test parameter validation and exception handling for bucket_by_sequence_length."""
    
    # Create a simple dataset
    dataset = create_variable_length_sequences(10, max_length=10)
    
    if invalid_scenario == "mismatched_batch_sizes":
        # bucket_batch_sizes length doesn't match bucket_boundaries + 1
        bucket_boundaries = [5, 10]
        bucket_batch_sizes = [2, 2]  # Should be length 3 (2 boundaries + 1)
        
        # Weak assertion 1: exception_raised
        with pytest.raises(expected_exception) as exc_info:
            transform_fn = bucket_by_sequence_length(
                element_length_func=element_length_func,
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes
            )
            _ = transform_fn(dataset)
        
        # Weak assertion 2: exception_type
        assert exc_info.type == expected_exception, f"Expected {expected_exception}, got {exc_info.type}"
        
        # Weak assertion 3: error_message_contains
        error_msg = str(exc_info.value).lower()
        # Check for relevant error indicators
        assert any(keyword in error_msg for keyword in ['length', 'size', 'boundary', 'batch']), \
            f"Error message should mention length/size/boundary/batch, got: {error_msg}"
    
    elif invalid_scenario == "non_increasing_boundaries":
        # bucket_boundaries is not strictly increasing
        bucket_boundaries = [10, 5, 15]  # Not increasing: 10 > 5
        bucket_batch_sizes = [2, 2, 2, 2]
        
        # Weak assertion 1: exception_raised
        with pytest.raises(expected_exception) as exc_info:
            transform_fn = bucket_by_sequence_length(
                element_length_func=element_length_func,
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes
            )
            _ = transform_fn(dataset)
        
        # Weak assertion 2: exception_type
        assert exc_info.type == expected_exception, f"Expected {expected_exception}, got {exc_info.type}"
        
        # Weak assertion 3: error_message_contains
        error_msg = str(exc_info.value).lower()
        # Check for relevant error indicators
        assert any(keyword in error_msg for keyword in ['increasing', 'strictly', 'order', 'boundary']), \
            f"Error message should mention increasing/strictly/order/boundary, got: {error_msg}"
    
    else:
        pytest.fail(f"Unknown invalid_scenario: {invalid_scenario}")
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# TC-07: bucket_by_sequence_length 填充选项 (DEFERRED)
# This test case is deferred and will be implemented in later iterations.
# Placeholder for bucket_by_sequence_length padding options test.
pass
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# TC-08: bucket_by_sequence_length 边界序列 (DEFERRED)
# This test case is deferred and will be implemented in later iterations.
# Placeholder for bucket_by_sequence_length edge cases test.
pass
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Additional test utilities and cleanup for G2

if __name__ == "__main__":
    # Simple test runner for debugging
    import sys
    pytest.main([sys.argv[0], "-v"])
# ==== BLOCK:FOOTER END ====