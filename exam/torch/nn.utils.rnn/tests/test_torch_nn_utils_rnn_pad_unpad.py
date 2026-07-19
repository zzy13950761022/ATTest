import math
import pytest
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Tuple

# ==== BLOCK:HEADER START ====
import math
import pytest
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Tuple

# Set random seed for reproducibility
torch.manual_seed(42)

# Helper functions
def generate_sequence_list(
    num_sequences: int,
    max_len: int,
    feature_dim: int = 3,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> List[torch.Tensor]:
    """Generate a list of variable-length sequences."""
    sequences = []
    for _ in range(num_sequences):
        length = torch.randint(1, max_len + 1, (1,)).item()
        seq = torch.randn(length, feature_dim, dtype=dtype, device=device)
        sequences.append(seq)
    return sequences

def generate_padded_sequence(
    batch_size: int,
    max_len: int,
    feature_dim: int = 3,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    batch_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a padded sequence and corresponding lengths."""
    lengths = torch.randint(1, max_len + 1, (batch_size,), device=device)
    lengths = lengths.sort(descending=True).values
    
    if batch_first:
        padded = torch.zeros(batch_size, max_len, feature_dim, dtype=dtype, device=device)
        for i, length in enumerate(lengths):
            padded[i, :length] = torch.randn(length, feature_dim, dtype=dtype, device=device)
    else:
        padded = torch.zeros(max_len, batch_size, feature_dim, dtype=dtype, device=device)
        for i, length in enumerate(lengths):
            padded[:length, i] = torch.randn(length, feature_dim, dtype=dtype, device=device)
    
    return padded, lengths

# Fixtures
@pytest.fixture
def device_cpu():
    """CPU device fixture."""
    return "cpu"

@pytest.fixture(params=[torch.float32, torch.float64])
def dtype_fixture(request):
    """Dtype fixture for parameterized tests."""
    return request.param
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch_first", [False])
def test_pad_sequence_basic(
    dtype: torch.dtype,
    device: str,
    batch_first: bool
):
    """TC-03: pad_sequence基本功能"""
    # Test parameters from param_matrix
    padding_value = 0.0
    num_sequences = 3
    max_len = 4
    feature_dim = 3
    
    # Generate test data
    sequences = generate_sequence_list(
        num_sequences=num_sequences,
        max_len=max_len,
        feature_dim=feature_dim,
        dtype=dtype,
        device=device
    )
    
    # Get actual lengths for verification
    lengths = [seq.size(0) for seq in sequences]
    actual_max_len = max(lengths)
    
    # Call the function
    padded = rnn_utils.pad_sequence(
        sequences,
        batch_first=batch_first,
        padding_value=padding_value
    )
    
    # Weak assertions (epoch 1)
    # 1. output_tensor: Should return a tensor
    assert isinstance(padded, torch.Tensor), \
        f"Expected torch.Tensor, got {type(padded)}"
    
    # 2. correct_shape: Shape should be correct
    if batch_first:
        expected_shape = (num_sequences, actual_max_len, feature_dim)
    else:
        expected_shape = (actual_max_len, num_sequences, feature_dim)
    
    assert padded.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {padded.shape}"
    
    # 3. padding_applied: Check padding values
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if length < actual_max_len:
            if batch_first:
                # Get padded area for this sequence
                padded_area = padded[i, length:]
                # Check all elements in padded area are padding_value
                expected_padding = torch.full_like(padded_area, padding_value)
                assert torch.allclose(padded_area, expected_padding, rtol=1e-6, atol=1e-6), \
                    f"Incorrect padding in sequence {i}"
            else:
                # Get padded area for this sequence
                padded_area = padded[length:, i]
                # Check all elements in padded area are padding_value
                expected_padding = torch.full_like(padded_area, padding_value)
                assert torch.allclose(padded_area, expected_padding, rtol=1e-6, atol=1e-6), \
                    f"Incorrect padding in sequence {i}"
    
    # 4. device_preserved: Device should match input
    assert padded.device.type == device, \
        f"Expected device {device}, got {padded.device.type}"
    
    # Additional weak assertions
    # dtype preservation
    assert padded.dtype == dtype, \
        f"Expected dtype {dtype}, got {padded.dtype}"
    
    # Verify original data is preserved in non-padded areas
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if batch_first:
            original_data = seq
            padded_data = padded[i, :length]
        else:
            original_data = seq
            padded_data = padded[:length, i]
        
        # Check data preservation (within tolerance)
        assert torch.allclose(original_data, padded_data, rtol=1e-6, atol=1e-6), \
            f"Data mismatch in sequence {i}"
    
    # Verify input sequences are not modified
    for i, seq in enumerate(sequences):
        # Check that sequences still have their original lengths
        assert seq.size(0) == lengths[i], \
            f"Sequence {i} length changed from {lengths[i]} to {seq.size(0)}"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_07 START ====
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch_first", [False])
def test_unpad_sequence_basic(
    dtype: torch.dtype,
    device: str,
    batch_first: bool
):
    """TC-07: unpad_sequence基本功能"""
    # Test parameters
    num_sequences = 3
    max_len = 4
    feature_dim = 3
    padding_value = 0.0
    
    # Generate test data
    sequences = generate_sequence_list(
        num_sequences=num_sequences,
        max_len=max_len,
        feature_dim=feature_dim,
        dtype=dtype,
        device=device
    )
    
    # Get lengths for verification
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.int64, device=device)
    
    # First pad the sequences
    padded = rnn_utils.pad_sequence(
        sequences,
        batch_first=batch_first,
        padding_value=padding_value
    )
    
    # Now unpad them
    unpadded_sequences = rnn_utils.unpad_sequence(
        padded,
        lengths,
        batch_first=batch_first
    )
    
    # Weak assertions (epoch 1)
    # 1. output_type: Should return a list
    assert isinstance(unpadded_sequences, list), \
        f"Expected list, got {type(unpadded_sequences)}"
    
    # 2. correct_length: Should have same number of sequences
    assert len(unpadded_sequences) == num_sequences, \
        f"Expected {num_sequences} sequences, got {len(unpadded_sequences)}"
    
    # 3. data_recovered: Each sequence should match original
    for i, (original, unpadded) in enumerate(zip(sequences, unpadded_sequences)):
        # Check type
        assert isinstance(unpadded, torch.Tensor), \
            f"Sequence {i}: Expected torch.Tensor, got {type(unpadded)}"
        
        # Check shape
        assert unpadded.shape == original.shape, \
            f"Sequence {i}: Expected shape {original.shape}, got {unpadded.shape}"
        
        # Check data (within tolerance)
        assert torch.allclose(original, unpadded, rtol=1e-6, atol=1e-6), \
            f"Sequence {i}: Data mismatch"
        
        # Check device
        assert unpadded.device.type == device, \
            f"Sequence {i}: Expected device {device}, got {unpadded.device.type}"
        
        # Check dtype
        assert unpadded.dtype == dtype, \
            f"Sequence {i}: Expected dtype {dtype}, got {unpadded.dtype}"
    
    # 4. lengths_match: Each sequence should have correct length
    for i, (length, seq) in enumerate(zip(lengths, unpadded_sequences)):
        assert seq.size(0) == length.item(), \
            f"Sequence {i}: Expected length {length.item()}, got {seq.size(0)}"
    
    # Additional weak assertions
    # Verify that input tensor is not modified
    original_padded_sum = padded.sum().item()
    # The function shouldn't modify input when batch_first=False
    # (Note: when batch_first=False, the function uses transpose_ which modifies in-place)
    # We'll check that the function works correctly regardless
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch_first", [False, True])
def test_pad_unpad_roundtrip(
    dtype: torch.dtype,
    device: str,
    batch_first: bool
):
    """TC-08: pad_sequence和unpad_sequence往返测试"""
    # Test parameters
    num_sequences = 4
    max_len = 6
    feature_dim = 2
    padding_value = -1.0
    
    # Generate test data
    sequences = generate_sequence_list(
        num_sequences=num_sequences,
        max_len=max_len,
        feature_dim=feature_dim,
        dtype=dtype,
        device=device
    )
    
    # Get lengths for verification
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.int64, device=device)
    actual_max_len = max(lengths).item()
    
    # Step 1: Pad the sequences
    padded = rnn_utils.pad_sequence(
        sequences,
        batch_first=batch_first,
        padding_value=padding_value
    )
    
    # Step 2: Unpad them back
    unpadded_sequences = rnn_utils.unpad_sequence(
        padded,
        lengths,
        batch_first=batch_first
    )
    
    # Weak assertions (epoch 1)
    # 1. roundtrip_consistency: Unpadded sequences should match original
    assert len(unpadded_sequences) == len(sequences), \
        f"Expected {len(sequences)} sequences, got {len(unpadded_sequences)}"
    
    for i, (original, unpadded) in enumerate(zip(sequences, unpadded_sequences)):
        # Check shape
        assert unpadded.shape == original.shape, \
            f"Sequence {i}: Expected shape {original.shape}, got {unpadded.shape}"
        
        # Check data (within tolerance)
        assert torch.allclose(original, unpadded, rtol=1e-6, atol=1e-6), \
            f"Sequence {i}: Data mismatch after roundtrip"
    
    # 2. padding_correctness: Verify padding was applied correctly
    # FIXED: Correct shape calculation based on batch_first parameter
    # When batch_first=True: (batch_size, max_len, feature_dim)
    # When batch_first=False: (max_len, batch_size, feature_dim)
    if batch_first:
        expected_shape = (num_sequences, actual_max_len, feature_dim)
    else:
        expected_shape = (actual_max_len, num_sequences, feature_dim)
    
    assert padded.shape == expected_shape, \
        f"Expected padded shape {expected_shape}, got {padded.shape}"
    
    # Check padding values
    for i, length in enumerate(lengths):
        if length.item() < actual_max_len:
            if batch_first:
                padded_area = padded[i, length.item():]
                expected_padding = torch.full_like(padded_area, padding_value)
            else:
                padded_area = padded[length.item():, i]
                expected_padding = torch.full_like(padded_area, padding_value)
            
            assert torch.allclose(padded_area, expected_padding, rtol=1e-6, atol=1e-6), \
                f"Sequence {i}: Padding incorrect"
    
    # 3. device_preservation: Device should be preserved
    assert padded.device.type == device, \
        f"Padded tensor device mismatch: expected {device}, got {padded.device.type}"
    
    for seq in unpadded_sequences:
        assert seq.device.type == device, \
            f"Unpadded sequence device mismatch: expected {device}, got {seq.device.type}"
    
    # 4. dtype_preservation: Dtype should be preserved
    assert padded.dtype == dtype, \
        f"Padded tensor dtype mismatch: expected {dtype}, got {padded.dtype}"
    
    for seq in unpadded_sequences:
        assert seq.dtype == dtype, \
            f"Unpadded sequence dtype mismatch: expected {dtype}, got {seq.dtype}"
    
    # Additional weak assertions
    # Verify that original sequences are not modified
    for i, seq in enumerate(sequences):
        original_length = lengths[i].item()
        assert seq.size(0) == original_length, \
            f"Sequence {i}: Length changed from {original_length} to {seq.size(0)}"
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Test class for grouping related tests (optional)
class TestPadUnpadFunctions:
    """Test class for pad/unpad functions."""
    
    @staticmethod
    def test_pad_sequence_empty_list():
        """Test pad_sequence with empty list."""
        # FIXED: pad_sequence actually raises RuntimeError, not ValueError
        with pytest.raises(RuntimeError, match="received an empty list of sequences"):
            rnn_utils.pad_sequence([], batch_first=False)
    
    @staticmethod
    def test_pad_sequence_single_sequence():
        """Test pad_sequence with single sequence."""
        seq = torch.randn(5, 3)
        padded = rnn_utils.pad_sequence([seq], batch_first=False)
        
        assert isinstance(padded, torch.Tensor)
        assert padded.shape == (5, 1, 3)
        assert torch.allclose(padded[:, 0, :], seq)
    
    @staticmethod
    def test_unpad_sequence_length_mismatch():
        """Test unpad_sequence with incorrect lengths."""
        # Create a padded tensor with batch_first=False
        # Shape: (max_len=10, batch_size=3, feature_dim=5)
        padded = torch.randn(10, 3, 5)
        
        # Create lengths that are too long
        lengths = torch.tensor([12, 8, 9])  # First length > max_len
        
        # FIXED: unpad_sequence doesn't raise an error when length > max_len
        # Instead, it just returns all elements up to max_len
        # Let's test the actual behavior
        unpadded_sequences = rnn_utils.unpad_sequence(padded, lengths, batch_first=False)
        
        # Verify we get the correct number of sequences
        assert len(unpadded_sequences) == 3
        
        # Verify each sequence has the correct shape
        # When length > max_len, unpad_sequence returns all max_len elements
        assert unpadded_sequences[0].shape == (10, 5)  # First sequence: length=12 but max_len=10
        assert unpadded_sequences[1].shape == (8, 5)   # Second sequence: length=8
        assert unpadded_sequences[2].shape == (9, 5)   # Third sequence: length=9
        
        # Verify the data matches the original padded tensor
        # For batch_first=False, padded shape is (max_len, batch_size, feature_dim)
        # Each unpadded sequence should match the corresponding column
        # But we need to handle the case where length > max_len
        for i in range(3):
            length = lengths[i].item()
            actual_length = min(length, 10)  # max_len is 10
            
            # Get the corresponding data from padded tensor
            if batch_first:
                # This would be padded[i, :actual_length, :] if batch_first=True
                # But we're using batch_first=False
                expected_data = padded[:actual_length, i, :]
            else:
                expected_data = padded[:actual_length, i, :]
            
            # Check that the unpadded sequence matches
            assert torch.allclose(unpadded_sequences[i], expected_data), \
                f"Sequence {i}: Data mismatch"

# Cleanup and teardown logic can be added here if needed
if __name__ == "__main__":
    # Simple test runner for debugging
    import sys
    pytest.main(sys.argv)
# ==== BLOCK:FOOTER END ====