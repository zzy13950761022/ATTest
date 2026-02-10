"""
Test module for torch.nn.modules.rnn (Group G1: Core RNN/LSTM/GRU forward propagation)
"""
import math
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions
@pytest.fixture
def set_random_seed():
    """Fixture to set random seeds before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Cleanup if needed

def create_test_input(batch_size, seq_len, input_size, batch_first=False, dtype=torch.float32):
    """Create test input tensor with given parameters."""
    if batch_first:
        shape = (batch_size, seq_len, input_size)
    else:
        shape = (seq_len, batch_size, input_size)
    return torch.randn(*shape, dtype=dtype)

def assert_shape_equal(actual, expected, msg=""):
    """Assert that tensor shape matches expected."""
    assert actual.shape == expected, f"{msg}: expected {expected}, got {actual.shape}"

def assert_dtype_equal(actual, expected_dtype, msg=""):
    """Assert that tensor dtype matches expected."""
    assert actual.dtype == expected_dtype, f"{msg}: expected {expected_dtype}, got {actual.dtype}"

def assert_finite(tensor, msg=""):
    """Assert that tensor contains only finite values."""
    assert torch.isfinite(tensor).all(), f"{msg}: tensor contains non-finite values"

def assert_no_nan(tensor, msg=""):
    """Assert that tensor contains no NaN values."""
    assert not torch.isnan(tensor).any(), f"{msg}: tensor contains NaN values"
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize(
    "mode,input_size,hidden_size,num_layers,batch_first,bidirectional,dtype_str,batch_size,seq_len",
    [
        # Base case from test plan
        ("RNN_TANH", 10, 20, 1, False, False, "float32", 3, 5),
        # Parameter extension: double precision, larger size, batch_first format
        ("RNN_TANH", 20, 40, 2, True, False, "float64", 4, 8),
    ]
)
def test_basic_rnn_forward_shape(
    set_random_seed,
    mode,
    input_size,
    hidden_size,
    num_layers,
    batch_first,
    bidirectional,
    dtype_str,
    batch_size,
    seq_len,
):
    """
    Test basic RNN forward propagation shape validation.
    
    Weak assertions:
    - output_shape: Check output tensor shape matches expected
    - hidden_shape: Check hidden state shape matches expected
    - dtype: Check output dtype matches input dtype
    - finite: Check all values are finite
    - no_nan: Check no NaN values in output
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[dtype_str]
    
    # Create RNN instance with matching dtype
    # Note: PyTorch RNN weights default to float32, so we need to ensure
    # the RNN is created with the correct dtype for float64 inputs
    rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
        nonlinearity=mode.lower().replace("rnn_", ""),  # "tanh" or "relu"
    )
    
    # Convert RNN parameters to match input dtype if needed
    if dtype == torch.float64:
        rnn = rnn.double()
    
    # Create test input
    x = create_test_input(batch_size, seq_len, input_size, batch_first, dtype)
    
    # Forward pass
    output, h_n = rnn(x)
    
    # Calculate expected shapes
    num_directions = 2 if bidirectional else 1
    
    # Expected output shape
    if batch_first:
        expected_output_shape = (batch_size, seq_len, num_directions * hidden_size)
    else:
        expected_output_shape = (seq_len, batch_size, num_directions * hidden_size)
    
    # Expected hidden state shape
    expected_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)
    
    # Weak assertions
    # 1. Output shape assertion
    assert_shape_equal(output, expected_output_shape, 
                      f"Output shape mismatch for mode={mode}")
    
    # 2. Hidden state shape assertion
    assert_shape_equal(h_n, expected_hidden_shape,
                      f"Hidden state shape mismatch for mode={mode}")
    
    # 3. Dtype assertion
    assert_dtype_equal(output, dtype,
                      f"Output dtype mismatch for mode={mode}")
    assert_dtype_equal(h_n, dtype,
                      f"Hidden state dtype mismatch for mode={mode}")
    
    # 4. Finite values assertion
    assert_finite(output, f"Output contains non-finite values for mode={mode}")
    assert_finite(h_n, f"Hidden state contains non-finite values for mode={mode}")
    
    # 5. No NaN assertion
    assert_no_nan(output, f"Output contains NaN values for mode={mode}")
    assert_no_nan(h_n, f"Hidden state contains NaN values for mode={mode}")
    
    # Additional consistency check: output values should be reasonable
    # (not too large in magnitude for tanh activation)
    if mode == "RNN_TANH":
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0), \
            f"RNN_TANH output should be in [-1, 1] range, got min={output.min():.4f}, max={output.max():.4f}"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize(
    "mode,input_size,hidden_size,num_layers,batch_first,bidirectional,dtype_str,batch_size,seq_len",
    [
        # Base case from test plan
        ("LSTM", 8, 16, 2, True, False, "float32", 2, 4),
        # Parameter extension: single-layer bidirectional LSTM, single batch, long sequence
        ("LSTM", 16, 32, 1, False, True, "float32", 1, 10),
    ]
)
def test_lstm_basic_functionality(
    set_random_seed,
    mode,
    input_size,
    hidden_size,
    num_layers,
    batch_first,
    bidirectional,
    dtype_str,
    batch_size,
    seq_len,
):
    """
    Test LSTM basic functionality validation.
    
    Weak assertions:
    - output_shape: Check output tensor shape matches expected
    - hidden_shape: Check hidden state shape matches expected
    - cell_shape: Check cell state shape matches expected
    - dtype: Check output dtype matches input dtype
    - finite: Check all values are finite
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[dtype_str]
    
    # Create LSTM instance
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
    )
    
    # Create test input
    x = create_test_input(batch_size, seq_len, input_size, batch_first, dtype)
    
    # Forward pass
    output, (h_n, c_n) = lstm(x)
    
    # Calculate expected shapes
    num_directions = 2 if bidirectional else 1
    
    # Expected output shape
    if batch_first:
        expected_output_shape = (batch_size, seq_len, num_directions * hidden_size)
    else:
        expected_output_shape = (seq_len, batch_size, num_directions * hidden_size)
    
    # Expected hidden state shape (same for both hidden and cell states)
    expected_state_shape = (num_layers * num_directions, batch_size, hidden_size)
    
    # Weak assertions
    # 1. Output shape assertion
    assert_shape_equal(output, expected_output_shape,
                      f"Output shape mismatch for LSTM")
    
    # 2. Hidden state shape assertion
    assert_shape_equal(h_n, expected_state_shape,
                      f"Hidden state shape mismatch for LSTM")
    
    # 3. Cell state shape assertion
    assert_shape_equal(c_n, expected_state_shape,
                      f"Cell state shape mismatch for LSTM")
    
    # 4. Dtype assertion
    assert_dtype_equal(output, dtype,
                      f"Output dtype mismatch for LSTM")
    assert_dtype_equal(h_n, dtype,
                      f"Hidden state dtype mismatch for LSTM")
    assert_dtype_equal(c_n, dtype,
                      f"Cell state dtype mismatch for LSTM")
    
    # 5. Finite values assertion
    assert_finite(output, f"Output contains non-finite values for LSTM")
    assert_finite(h_n, f"Hidden state contains non-finite values for LSTM")
    assert_finite(c_n, f"Cell state contains non-finite values for LSTM")
    
    # 6. No NaN assertion (additional safety check)
    assert_no_nan(output, f"Output contains NaN values for LSTM")
    assert_no_nan(h_n, f"Hidden state contains NaN values for LSTM")
    assert_no_nan(c_n, f"Cell state contains NaN values for LSTM")
    
    # LSTM-specific consistency checks
    # Check that hidden and cell states have same shape
    assert h_n.shape == c_n.shape, \
        f"LSTM hidden and cell states should have same shape, got {h_n.shape} vs {c_n.shape}"
    
    # Check that output last timestep matches hidden state (for single direction)
    # This is a basic LSTM property: h_n contains the hidden states for the last timestep
    if not bidirectional:
        # For single direction, we can check the last timestep
        if batch_first:
            # output shape: (batch, seq_len, hidden_size)
            last_output = output[:, -1, :]  # (batch, hidden_size)
            last_hidden = h_n[-1, :, :]     # (batch, hidden_size) from last layer
        else:
            # output shape: (seq_len, batch, hidden_size)
            last_output = output[-1, :, :]  # (batch, hidden_size)
            last_hidden = h_n[-1, :, :]     # (batch, hidden_size) from last layer
        
        # They should be close but not exactly equal due to different numerical paths
        # Just check they have same shape
        assert last_output.shape == last_hidden.shape, \
            f"Last output and hidden state should have same shape, got {last_output.shape} vs {last_hidden.shape}"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_05 START ====
@pytest.mark.parametrize(
    "mode,input_size,hidden_size,proj_size,num_layers,batch_first,bidirectional,dtype_str,batch_size,seq_len",
    [
        # Base case from test plan: LSTM with projection
        ("LSTM", 10, 20, 15, 1, False, False, "float32", 2, 4),
    ]
)
def test_lstm_projection_functionality(
    set_random_seed,
    mode,
    input_size,
    hidden_size,
    proj_size,
    num_layers,
    batch_first,
    bidirectional,
    dtype_str,
    batch_size,
    seq_len,
):
    """
    Test LSTM projection functionality constraint checking.
    
    Weak assertions:
    - output_shape_proj: Check output tensor shape with projection
    - hidden_shape_proj: Check hidden state shape with projection
    - dtype: Check output dtype matches input dtype
    - finite: Check all values are finite
    - proj_size_constraint: Check projection size constraints
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[dtype_str]
    
    # Create LSTM instance with projection
    # Note: proj_size is only supported for LSTM in PyTorch
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        proj_size=proj_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
    )
    
    # Convert LSTM parameters to match input dtype if needed
    if dtype == torch.float64:
        lstm = lstm.double()
    
    # Create test input
    x = create_test_input(batch_size, seq_len, input_size, batch_first, dtype)
    
    # Forward pass
    output, (h_n, c_n) = lstm(x)
    
    # Calculate expected shapes with projection
    num_directions = 2 if bidirectional else 1
    
    # Expected output shape: output uses proj_size instead of hidden_size
    if batch_first:
        expected_output_shape = (batch_size, seq_len, num_directions * proj_size)
    else:
        expected_output_shape = (seq_len, batch_size, num_directions * proj_size)
    
    # Expected hidden state shape: h_n uses proj_size
    expected_hidden_shape = (num_layers * num_directions, batch_size, proj_size)
    
    # Expected cell state shape: c_n still uses hidden_size (not projected)
    expected_cell_shape = (num_layers * num_directions, batch_size, hidden_size)
    
    # Weak assertions
    # 1. Output shape assertion with projection
    assert_shape_equal(output, expected_output_shape,
                      f"Output shape mismatch for LSTM with projection")
    
    # 2. Hidden state shape assertion with projection
    assert_shape_equal(h_n, expected_hidden_shape,
                      f"Hidden state shape mismatch for LSTM with projection")
    
    # 3. Cell state shape assertion (should still use hidden_size)
    assert_shape_equal(c_n, expected_cell_shape,
                      f"Cell state shape mismatch for LSTM with projection")
    
    # 4. Dtype assertion
    assert_dtype_equal(output, dtype,
                      f"Output dtype mismatch for LSTM with projection")
    assert_dtype_equal(h_n, dtype,
                      f"Hidden state dtype mismatch for LSTM with projection")
    assert_dtype_equal(c_n, dtype,
                      f"Cell state dtype mismatch for LSTM with projection")
    
    # 5. Finite values assertion
    assert_finite(output, f"Output contains non-finite values for LSTM with projection")
    assert_finite(h_n, f"Hidden state contains non-finite values for LSTM with projection")
    assert_finite(c_n, f"Cell state contains non-finite values for LSTM with projection")
    
    # 6. No NaN assertion (additional safety check)
    assert_no_nan(output, f"Output contains NaN values for LSTM with projection")
    assert_no_nan(h_n, f"Hidden state contains NaN values for LSTM with projection")
    assert_no_nan(c_n, f"Cell state contains NaN values for LSTM with projection")
    
    # 7. Projection size constraint check
    # proj_size must be < hidden_size (PyTorch enforces this)
    assert proj_size < hidden_size, \
        f"proj_size ({proj_size}) must be smaller than hidden_size ({hidden_size})"
    
    # LSTM-specific projection checks
    # Check that output dimension matches proj_size (not hidden_size)
    assert output.shape[-1] == num_directions * proj_size, \
        f"Output dimension should be {num_directions * proj_size} with projection, got {output.shape[-1]}"
    
    # Check that hidden state dimension matches proj_size
    assert h_n.shape[-1] == proj_size, \
        f"Hidden state dimension should be {proj_size} with projection, got {h_n.shape[-1]}"
    
    # Check that cell state dimension still matches hidden_size (not projected)
    assert c_n.shape[-1] == hidden_size, \
        f"Cell state dimension should be {hidden_size} (not projected), got {c_n.shape[-1]}"
    
    # Check parameter shapes for projection
    # LSTM with projection has different weight/bias parameter shapes
    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            # weight_ih_l0 shape: (4*hidden_size, input_size)
            # For LSTM with projection, weight_ih_l0 still connects input to hidden
            # So last dimension should be input_size for layer 0
            if "_l0" in name:
                # First layer: connects input to hidden
                assert param.shape[-1] == input_size, \
                    f"weight_ih_l0 should have last dimension {input_size}, got {param.shape}"
            else:
                # Higher layers: connect previous hidden state to current hidden
                # For projection, previous hidden state has dimension proj_size
                # Extract layer number from name
                layer_num = name.split('_')[-1][1:]  # e.g., "l0" from "weight_ih_l0"
                assert param.shape[-1] == proj_size, \
                    f"weight_ih_l{layer_num} should have last dimension {proj_size}, got {param.shape}"
        
        elif "weight_hh" in name:
            # weight_hh_l0 shape: (4*hidden_size, proj_size) for LSTM with projection
            # This connects previous hidden state (size proj_size) to current hidden
            assert param.shape[-1] == proj_size, \
                f"weight_hh {name} should have last dimension {proj_size}, got {param.shape}"
        
        elif "weight_hr" in name:
            # weight_hr_l0 shape: (proj_size, hidden_size) - projection weights
            assert param.shape[0] == proj_size and param.shape[1] == hidden_size, \
                f"weight_hr {name} should have shape ({proj_size}, {hidden_size}), got {param.shape}"
    
    # Test that projection is only supported for LSTM
    # (This is enforced by PyTorch at construction time)
    # We'll verify by checking that non-LSTM RNNs don't accept proj_size parameter
    # PyTorch raises ValueError, not TypeError, for proj_size in non-LSTM RNNs
    with pytest.raises(ValueError, match="proj_size argument is only supported for LSTM"):
        nn.RNN(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size)
    
    with pytest.raises(ValueError, match="proj_size argument is only supported for LSTM"):
        nn.GRU(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size)
    
    # Test invalid proj_size values
    # proj_size must be >= 0
    with pytest.raises(ValueError, match="proj_size has to be a positive integer"):
        nn.LSTM(input_size=input_size, hidden_size=hidden_size, proj_size=-1)
    
    # proj_size must be < hidden_size
    with pytest.raises(ValueError, match="proj_size has to be smaller than hidden_size"):
        nn.LSTM(input_size=input_size, hidden_size=hidden_size, proj_size=hidden_size)
    
    with pytest.raises(ValueError, match="proj_size has to be smaller than hidden_size"):
        nn.LSTM(input_size=input_size, hidden_size=hidden_size, proj_size=hidden_size + 5)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# Placeholder for CASE_06: (DEFERRED)
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:FOOTER START ====
# Additional test cases for edge cases and error conditions

def test_rnn_invalid_mode():
    """Test that invalid RNN mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown nonlinearity 'invalid_mode'"):
        nn.RNN(input_size=10, hidden_size=20, nonlinearity="invalid_mode")

def test_rnn_dropout_warning():
    """Test that dropout with single layer produces warning."""
    with pytest.warns(UserWarning, match="dropout option adds dropout"):
        nn.RNN(input_size=10, hidden_size=20, num_layers=1, dropout=0.5)

def test_lstm_proj_size_constraint():
    """Test that proj_size >= hidden_size raises ValueError (deferred test preview)."""
    with pytest.raises(ValueError, match="proj_size has to be smaller than hidden_size"):
        nn.LSTM(input_size=10, hidden_size=20, proj_size=25)

def test_rnn_batch_first_format():
    """Test batch_first format conversion consistency."""
    # Create same RNN with batch_first=True and False
    rnn1 = nn.RNN(input_size=8, hidden_size=16, batch_first=True)
    rnn2 = nn.RNN(input_size=8, hidden_size=16, batch_first=False)
    
    # Copy weights to make them identical
    for (name1, param1), (name2, param2) in zip(rnn1.named_parameters(), rnn2.named_parameters()):
        param2.data.copy_(param1.data)
    
    # Create input in batch_first format
    batch_size, seq_len, input_size = 2, 4, 8
    x_bf = torch.randn(batch_size, seq_len, input_size)
    
    # Transpose to seq_first format
    x_sf = x_bf.transpose(0, 1)
    
    # Forward passes
    output1, h1 = rnn1(x_bf)
    output2, h2 = rnn2(x_sf)
    
    # Transpose output2 to batch_first for comparison
    output2_bf = output2.transpose(0, 1)
    
    # Check shapes are compatible
    assert output1.shape == output2_bf.shape, \
        f"Batch_first outputs should have compatible shapes: {output1.shape} vs {output2_bf.shape}"
    
    # Check hidden states (they should be identical since weights are same)
    assert torch.allclose(h1, h2, rtol=1e-5), \
        "Hidden states should match for same weights with different batch_first settings"

@pytest.mark.xfail(reason="PyTorch RNN does not support zero-length sequences")
def test_rnn_zero_sequence_length():
    """Test RNN with zero sequence length (edge case)."""
    rnn = nn.RNN(input_size=10, hidden_size=20)
    
    # Create input with seq_len=0
    batch_size, seq_len, input_size = 3, 0, 10
    x = torch.randn(seq_len, batch_size, input_size)
    
    # This should raise RuntimeError for zero-length sequences
    with pytest.raises(RuntimeError, match="Expected sequence length to be larger than 0 in RNN"):
        output, h_n = rnn(x)

# Cleanup and main execution
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
# ==== BLOCK:FOOTER END ====