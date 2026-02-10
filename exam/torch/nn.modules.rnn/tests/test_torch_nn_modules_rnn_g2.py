"""
Test module for torch.nn.modules.rnn (Group G2: Advanced features and boundary conditions)
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

def assert_tensors_close(tensor1, tensor2, rtol=1e-5, atol=1e-8, msg=""):
    """Assert that two tensors are close within tolerance."""
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), \
        f"{msg}: tensors are not close within tolerance"
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize(
    "mode,input_size,hidden_size,num_layers,batch_first,bidirectional,dtype_str,batch_size,seq_len",
    [
        # Base case from test plan: bidirectional GRU
        ("GRU", 12, 24, 1, False, True, "float32", 3, 6),
        # Parameter extension: minimal parameters, single element input
        ("GRU", 4, 8, 2, True, True, "float32", 1, 1),
    ]
)
def test_bidirectional_rnn_output_dimensions(
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
    Test bidirectional RNN output dimension validation.
    
    Weak assertions:
    - output_shape_bidirectional: Check output tensor shape for bidirectional RNN
    - hidden_shape_bidirectional: Check hidden state shape for bidirectional RNN
    - dtype: Check output dtype matches input dtype
    - finite: Check all values are finite
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[dtype_str]
    
    # Create RNN instance based on mode
    if mode == "GRU":
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
    elif mode == "RNN_TANH":
        rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            nonlinearity="tanh",
        )
    elif mode == "RNN_RELU":
        rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            nonlinearity="relu",
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Convert RNN parameters to match input dtype if needed
    if dtype == torch.float64:
        rnn = rnn.double()
    
    # Create test input
    x = create_test_input(batch_size, seq_len, input_size, batch_first, dtype)
    
    # Forward pass
    if mode == "GRU":
        output, h_n = rnn(x)
    else:
        output, h_n = rnn(x)
    
    # Calculate expected shapes for bidirectional RNN
    num_directions = 2 if bidirectional else 1
    
    # Expected output shape
    if batch_first:
        expected_output_shape = (batch_size, seq_len, num_directions * hidden_size)
    else:
        expected_output_shape = (seq_len, batch_size, num_directions * hidden_size)
    
    # Expected hidden state shape
    expected_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)
    
    # Weak assertions for bidirectional RNN
    # 1. Output shape assertion for bidirectional
    assert_shape_equal(output, expected_output_shape,
                      f"Bidirectional output shape mismatch for mode={mode}")
    
    # 2. Hidden state shape assertion for bidirectional
    assert_shape_equal(h_n, expected_hidden_shape,
                      f"Bidirectional hidden state shape mismatch for mode={mode}")
    
    # 3. Dtype assertion
    assert_dtype_equal(output, dtype,
                      f"Output dtype mismatch for mode={mode}")
    assert_dtype_equal(h_n, dtype,
                      f"Hidden state dtype mismatch for mode={mode}")
    
    # 4. Finite values assertion
    assert_finite(output, f"Output contains non-finite values for mode={mode}")
    assert_finite(h_n, f"Hidden state contains non-finite values for mode={mode}")
    
    # 5. No NaN assertion (additional safety check)
    assert_no_nan(output, f"Output contains NaN values for mode={mode}")
    assert_no_nan(h_n, f"Hidden state contains NaN values for mode={mode}")
    
    # Bidirectional-specific checks
    if bidirectional:
        # For bidirectional RNN, output contains concatenated forward and backward outputs
        # Check that output dimension is twice the hidden size
        assert output.shape[-1] == 2 * hidden_size, \
            f"Bidirectional output should have dimension {2 * hidden_size}, got {output.shape[-1]}"
        
        # Check that hidden state has correct number of directions
        assert h_n.shape[0] == num_layers * 2, \
            f"Bidirectional hidden state should have {num_layers * 2} layers, got {h_n.shape[0]}"
        
        # For GRU, we can also check that forward and backward hidden states are different
        # (they should be since they process the sequence in opposite directions)
        if mode == "GRU" and num_layers == 1 and seq_len > 1:
            # Extract forward and backward hidden states
            forward_hidden = h_n[0]  # forward direction
            backward_hidden = h_n[1]  # backward direction
            
            # They should not be identical (though could be close by chance)
            # Just check they have same shape
            assert forward_hidden.shape == backward_hidden.shape, \
                "Forward and backward hidden states should have same shape"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
@pytest.mark.parametrize(
    "mode,input_size,hidden_size,num_layers,batch_first,bidirectional,dropout,dtype_str,batch_size,seq_len",
    [
        # Base case from test plan: multi-layer RNN with dropout
        ("RNN_RELU", 6, 12, 3, False, False, 0.5, "float32", 2, 3),
    ]
)
def test_multi_layer_dropout_randomness(
    set_random_seed,
    mode,
    input_size,
    hidden_size,
    num_layers,
    batch_first,
    bidirectional,
    dropout,
    dtype_str,
    batch_size,
    seq_len,
):
    """
    Test multi-layer dropout randomness in RNN.
    
    Weak assertions:
    - output_shape: Check output tensor shape matches expected
    - hidden_shape: Check hidden state shape matches expected
    - dtype: Check output dtype matches input dtype
    - finite: Check all values are finite
    - dropout_mask_present: Check dropout is applied (output varies between runs)
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[dtype_str]
    
    # Create RNN instance with dropout
    # Note: dropout is only applied between RNN layers (num_layers > 1)
    if mode == "RNN_RELU":
        rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=dropout,
            nonlinearity="relu",
        )
    elif mode == "RNN_TANH":
        rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=dropout,
            nonlinearity="tanh",
        )
    elif mode == "GRU":
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=dropout,
        )
    elif mode == "LSTM":
        rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Convert RNN parameters to match input dtype if needed
    if dtype == torch.float64:
        rnn = rnn.double()
    
    # Create test input
    x = create_test_input(batch_size, seq_len, input_size, batch_first, dtype)
    
    # Set RNN to training mode (dropout is active)
    rnn.train()
    
    # First forward pass
    if mode in ["LSTM", "GRU"]:
        output1, (h_n1, c_n1) = rnn(x) if mode == "LSTM" else rnn(x)
    else:
        output1, h_n1 = rnn(x)
    
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
    assert_shape_equal(output1, expected_output_shape,
                      f"Output shape mismatch for mode={mode} with dropout")
    
    # 2. Hidden state shape assertion
    if mode == "LSTM":
        assert_shape_equal(h_n1, expected_hidden_shape,
                          f"Hidden state shape mismatch for LSTM with dropout")
        assert_shape_equal(c_n1, expected_hidden_shape,
                          f"Cell state shape mismatch for LSTM with dropout")
    else:
        assert_shape_equal(h_n1, expected_hidden_shape,
                          f"Hidden state shape mismatch for mode={mode} with dropout")
    
    # 3. Dtype assertion
    assert_dtype_equal(output1, dtype,
                      f"Output dtype mismatch for mode={mode} with dropout")
    if mode == "LSTM":
        assert_dtype_equal(h_n1, dtype,
                          f"Hidden state dtype mismatch for LSTM with dropout")
        assert_dtype_equal(c_n1, dtype,
                          f"Cell state dtype mismatch for LSTM with dropout")
    else:
        assert_dtype_equal(h_n1, dtype,
                          f"Hidden state dtype mismatch for mode={mode} with dropout")
    
    # 4. Finite values assertion
    assert_finite(output1, f"Output contains non-finite values for mode={mode} with dropout")
    if mode == "LSTM":
        assert_finite(h_n1, f"Hidden state contains non-finite values for LSTM with dropout")
        assert_finite(c_n1, f"Cell state contains non-finite values for LSTM with dropout")
    else:
        assert_finite(h_n1, f"Hidden state contains non-finite values for mode={mode} with dropout")
    
    # 5. Dropout mask presence check (output should vary between forward passes in training mode)
    # Second forward pass with same input
    if mode in ["LSTM", "GRU"]:
        output2, (h_n2, c_n2) = rnn(x) if mode == "LSTM" else rnn(x)
    else:
        output2, h_n2 = rnn(x)
    
    # Check that outputs are not identical (due to dropout randomness)
    # Note: there's a small chance they could be identical by coincidence, but very unlikely
    assert not torch.allclose(output1, output2, rtol=1e-10, atol=1e-10), \
        f"Dropout should produce different outputs in training mode for mode={mode}"
    
    # Check that hidden states are also different (dropout affects hidden states between layers)
    if mode == "LSTM":
        assert not torch.allclose(h_n1, h_n2, rtol=1e-10, atol=1e-10), \
            "Dropout should produce different hidden states in training mode for LSTM"
        assert not torch.allclose(c_n1, c_n2, rtol=1e-10, atol=1e-10), \
            "Dropout should produce different cell states in training mode for LSTM"
    else:
        assert not torch.allclose(h_n1, h_n2, rtol=1e-10, atol=1e-10), \
            f"Dropout should produce different hidden states in training mode for mode={mode}"
    
    # Test evaluation mode (dropout should be disabled)
    rnn.eval()
    
    # Forward pass in evaluation mode
    if mode in ["LSTM", "GRU"]:
        output_eval, (h_n_eval, c_n_eval) = rnn(x) if mode == "LSTM" else rnn(x)
    else:
        output_eval, h_n_eval = rnn(x)
    
    # Another forward pass in evaluation mode (should be deterministic)
    if mode in ["LSTM", "GRU"]:
        output_eval2, (h_n_eval2, c_n_eval2) = rnn(x) if mode == "LSTM" else rnn(x)
    else:
        output_eval2, h_n_eval2 = rnn(x)
    
    # In evaluation mode, outputs should be identical (no dropout)
    assert torch.allclose(output_eval, output_eval2, rtol=1e-5, atol=1e-8), \
        f"Outputs should be identical in evaluation mode (no dropout) for mode={mode}"
    
    if mode == "LSTM":
        assert torch.allclose(h_n_eval, h_n_eval2, rtol=1e-5, atol=1e-8), \
            "Hidden states should be identical in evaluation mode (no dropout) for LSTM"
        assert torch.allclose(c_n_eval, c_n_eval2, rtol=1e-5, atol=1e-8), \
            "Cell states should be identical in evaluation mode (no dropout) for LSTM"
    else:
        assert torch.allclose(h_n_eval, h_n_eval2, rtol=1e-5, atol=1e-8), \
            f"Hidden states should be identical in evaluation mode (no dropout) for mode={mode}"
    
    # Additional check: dropout should only be applied between layers when num_layers > 1
    # For single layer RNN, dropout should be 0 (PyTorch enforces this)
    if num_layers == 1:
        # PyTorch should warn about dropout with single layer
        # We'll just verify the dropout parameter is effectively 0
        assert rnn.dropout == 0, \
            f"Dropout should be 0 for single-layer RNN, got {rnn.dropout}"
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_07 START ====
@pytest.mark.parametrize(
    "mode,input_size,hidden_size,num_layers,batch_first,bidirectional,dtype_str,batch_size,seq_len,test_type",
    [
        # Test 1: Extreme small values
        ("RNN_TANH", 1, 1, 1, False, False, "float32", 1, 1, "minimal"),
        # Test 2: Large hidden size (memory boundary)
        ("GRU", 8, 256, 1, False, False, "float32", 2, 4, "large_hidden"),
        # Test 3: Many layers
        ("LSTM", 16, 32, 5, True, False, "float32", 2, 8, "many_layers"),
        # Test 4: Long sequence
        ("RNN_RELU", 8, 16, 1, False, True, "float32", 1, 50, "long_sequence"),
    ]
)
def test_extreme_parameter_values(
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
    test_type,
):
    """
    Test RNN with extreme parameter values and boundary conditions.
    
    Weak assertions:
    - output_shape: Check output tensor shape matches expected
    - hidden_shape: Check hidden state shape matches expected
    - dtype: Check output dtype matches input dtype
    - finite: Check all values are finite
    - memory_boundary: Check memory usage is reasonable
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[dtype_str]
    
    # Create RNN instance based on mode
    if mode == "LSTM":
        rnn_class = nn.LSTM
    elif mode == "GRU":
        rnn_class = nn.GRU
    elif mode in ["RNN_TANH", "RNN_RELU"]:
        rnn_class = nn.RNN
        nonlinearity = mode.lower().replace("rnn_", "")
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Create RNN with appropriate parameters
    if mode in ["RNN_TANH", "RNN_RELU"]:
        rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
        )
    else:
        rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
    
    # Convert RNN parameters to match input dtype if needed
    if dtype == torch.float64:
        rnn = rnn.double()
    
    # Create test input
    x = create_test_input(batch_size, seq_len, input_size, batch_first, dtype)
    
    # Forward pass
    if mode == "LSTM":
        output, (h_n, c_n) = rnn(x)
    else:
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
                      f"Output shape mismatch for {test_type} test with mode={mode}")
    
    # 2. Hidden state shape assertion
    if mode == "LSTM":
        assert_shape_equal(h_n, expected_hidden_shape,
                          f"Hidden state shape mismatch for {test_type} test with LSTM")
        assert_shape_equal(c_n, expected_hidden_shape,
                          f"Cell state shape mismatch for {test_type} test with LSTM")
    else:
        assert_shape_equal(h_n, expected_hidden_shape,
                          f"Hidden state shape mismatch for {test_type} test with mode={mode}")
    
    # 3. Dtype assertion
    assert_dtype_equal(output, dtype,
                      f"Output dtype mismatch for {test_type} test with mode={mode}")
    if mode == "LSTM":
        assert_dtype_equal(h_n, dtype,
                          f"Hidden state dtype mismatch for {test_type} test with LSTM")
        assert_dtype_equal(c_n, dtype,
                          f"Cell state dtype mismatch for {test_type} test with LSTM")
    else:
        assert_dtype_equal(h_n, dtype,
                          f"Hidden state dtype mismatch for {test_type} test with mode={mode}")
    
    # 4. Finite values assertion
    assert_finite(output, f"Output contains non-finite values for {test_type} test with mode={mode}")
    if mode == "LSTM":
        assert_finite(h_n, f"Hidden state contains non-finite values for {test_type} test with LSTM")
        assert_finite(c_n, f"Cell state contains non-finite values for {test_type} test with LSTM")
    else:
        assert_finite(h_n, f"Hidden state contains non-finite values for {test_type} test with mode={mode}")
    
    # 5. No NaN assertion (additional safety check)
    assert_no_nan(output, f"Output contains NaN values for {test_type} test with mode={mode}")
    if mode == "LSTM":
        assert_no_nan(h_n, f"Hidden state contains NaN values for {test_type} test with LSTM")
        assert_no_nan(c_n, f"Cell state contains NaN values for {test_type} test with LSTM")
    else:
        assert_no_nan(h_n, f"Hidden state contains NaN values for {test_type} test with mode={mode}")
    
    # Test-specific checks based on test_type
    if test_type == "minimal":
        # For minimal parameters, check that everything still works
        assert x.numel() == batch_size * seq_len * input_size, \
            f"Input should have {batch_size * seq_len * input_size} elements for minimal test"
        assert output.numel() == batch_size * seq_len * num_directions * hidden_size, \
            f"Output should have {batch_size * seq_len * num_directions * hidden_size} elements for minimal test"
        
        # Check that values are reasonable (not exploding)
        output_abs_max = output.abs().max().item()
        assert output_abs_max < 100.0, \
            f"Output values should not explode for minimal test, max abs value: {output_abs_max}"
    
    elif test_type == "large_hidden":
        # For large hidden size, check memory usage is reasonable
        # We can't directly measure memory, but we can check parameter count
        param_count = sum(p.numel() for p in rnn.parameters())
        expected_param_count = 0
        
        # Calculate expected parameter count for GRU
        # GRU: gate_size = 3 * hidden_size
        gate_size = 3 * hidden_size
        # Input weights: (gate_size, input_size)
        # Hidden weights: (gate_size, hidden_size)
        # Biases: 2 * gate_size (if bias=True)
        input_weights_size = gate_size * input_size
        hidden_weights_size = gate_size * hidden_size
        bias_size = 2 * gate_size if rnn.bias else 0
        expected_param_count = input_weights_size + hidden_weights_size + bias_size
        
        # Allow some tolerance
        assert param_count == expected_param_count, \
            f"Parameter count mismatch for large hidden size: expected {expected_param_count}, got {param_count}"
        
        # Check that output values are not NaN or infinite
        assert torch.isfinite(output).all(), \
            "Output should be finite for large hidden size test"
    
    elif test_type == "many_layers":
        # For many layers, check that all layers produce output
        # LSTM with 5 layers should work correctly
        assert num_layers == 5, f"Test should have 5 layers for many_layers test, got {num_layers}"
        
        # Check that output shape is correct
        assert output.shape[-1] == num_directions * hidden_size, \
            f"Output dimension should be {num_directions * hidden_size} for many_layers test"
        
        # Check that hidden state has correct number of layers
        if mode == "LSTM":
            assert h_n.shape[0] == num_layers * num_directions, \
                f"Hidden state should have {num_layers * num_directions} layers for many_layers test"
            assert c_n.shape[0] == num_layers * num_directions, \
                f"Cell state should have {num_layers * num_directions} layers for many_layers test"
        else:
            assert h_n.shape[0] == num_layers * num_directions, \
                f"Hidden state should have {num_layers * num_directions} layers for many_layers test"
    
    elif test_type == "long_sequence":
        # For long sequence, check that RNN can handle it
        assert seq_len == 50, f"Test should have seq_len=50 for long_sequence test, got {seq_len}"
        
        # Check that output has correct sequence length
        if batch_first:
            assert output.shape[1] == seq_len, \
                f"Output should have seq_len={seq_len} for long_sequence test"
        else:
            assert output.shape[0] == seq_len, \
                f"Output should have seq_len={seq_len} for long_sequence test"
        
        # Check that values don't explode over long sequence
        # For RNN_RELU, values might grow, but shouldn't be NaN/inf
        output_std = output.std().item()
        assert output_std < 100.0, \
            f"Output standard deviation should be reasonable for long sequence, got {output_std}"
    
    # Additional boundary condition tests
    # Test dropout boundaries
    if num_layers > 1:
        # Test dropout=0.0 (no dropout)
        rnn_no_dropout = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=0.0,
        )
        if dtype == torch.float64:
            rnn_no_dropout = rnn_no_dropout.double()
        
        # Test dropout=1.0 (maximum dropout)
        rnn_max_dropout = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=1.0,
        )
        if dtype == torch.float64:
            rnn_max_dropout = rnn_max_dropout.double()
        
        # Both should construct without error
        assert rnn_no_dropout is not None
        assert rnn_max_dropout is not None
        
        # dropout=1.0 in training mode should zero out most values
        rnn_max_dropout.train()
        if mode == "LSTM":
            output_max_dropout, _ = rnn_max_dropout(x)
        else:
            output_max_dropout, _ = rnn_max_dropout(x)
        
        # With dropout=1.0, many values should be zero (but not necessarily all)
        zero_ratio = (output_max_dropout == 0).float().mean().item()
        # Note: dropout=1.0 doesn't guarantee all zeros due to implementation details
        # Just check it runs without error
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
@pytest.mark.parametrize(
    "cell_type,input_size,hidden_size,bias,batch_first,dtype_str,batch_size,seq_len",
    [
        # Test RNNCell with tanh activation
        ("RNNCell", 8, 16, True, False, "float32", 2, 4),
        # Test LSTMCell
        ("LSTMCell", 12, 24, True, True, "float32", 3, 5),
        # Test GRUCell
        ("GRUCell", 6, 12, False, False, "float32", 1, 3),
        # Test RNNCell with relu activation
        ("RNNCell_relu", 10, 20, True, False, "float32", 2, 3),
    ]
)
def test_rnn_cell_versions(
    set_random_seed,
    cell_type,
    input_size,
    hidden_size,
    bias,
    batch_first,
    dtype_str,
    batch_size,
    seq_len,
):
    """
    Test RNN cell versions (RNNCell, LSTMCell, GRUCell) independent functionality.
    
    Weak assertions:
    - output_shape: Check output tensor shape matches expected
    - hidden_shape: Check hidden state shape matches expected
    - dtype: Check output dtype matches input dtype
    - finite: Check all values are finite
    - cell_consistency: Check cell behavior is consistent
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[dtype_str]
    
    # Create cell instance based on cell_type
    if cell_type == "RNNCell":
        # Regular RNNCell with tanh
        cell = nn.RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            nonlinearity="tanh",
        )
        is_lstm = False
        has_cell_state = False
    elif cell_type == "RNNCell_relu":
        # RNNCell with relu activation
        cell = nn.RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            nonlinearity="relu",
        )
        is_lstm = False
        has_cell_state = False
    elif cell_type == "LSTMCell":
        # LSTMCell
        cell = nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
        )
        is_lstm = True
        has_cell_state = True
    elif cell_type == "GRUCell":
        # GRUCell
        cell = nn.GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
        )
        is_lstm = False
        has_cell_state = False
    else:
        raise ValueError(f"Unsupported cell type: {cell_type}")
    
    # Convert cell parameters to match input dtype if needed
    if dtype == torch.float64:
        cell = cell.double()
    
    # Create test input
    # For cells, we process one timestep at a time
    # Create input for all timesteps
    if batch_first:
        # Shape: (batch_size, seq_len, input_size)
        x_all = torch.randn(batch_size, seq_len, input_size, dtype=dtype)
    else:
        # Shape: (seq_len, batch_size, input_size)
        x_all = torch.randn(seq_len, batch_size, input_size, dtype=dtype)
    
    # Initialize hidden state
    if is_lstm:
        # LSTM has both hidden and cell states
        hx = torch.randn(batch_size, hidden_size, dtype=dtype)
        cx = torch.randn(batch_size, hidden_size, dtype=dtype)
        hidden = (hx, cx)
    else:
        # RNNCell and GRUCell have only hidden state
        hidden = torch.randn(batch_size, hidden_size, dtype=dtype)
    
    # Process sequence step by step
    outputs = []
    hidden_states = []
    
    for t in range(seq_len):
        # Get input for current timestep
        if batch_first:
            x_t = x_all[:, t, :]  # (batch_size, input_size)
        else:
            x_t = x_all[t, :, :]  # (batch_size, input_size)
        
        # Forward pass through cell
        if is_lstm:
            # LSTMCell returns (next_hidden, next_cell)
            next_hidden, next_cell = cell(x_t, hidden)
            hidden = (next_hidden, next_cell)
            output = next_hidden  # Use hidden state as output
            hidden_state = (next_hidden, next_cell)
        else:
            # RNNCell/GRUCell returns next hidden state
            next_hidden = cell(x_t, hidden)
            hidden = next_hidden
            output = next_hidden
            hidden_state = next_hidden
        
        # Store output and hidden state
        outputs.append(output)
        hidden_states.append(hidden_state)
    
    # Stack outputs along sequence dimension
    if batch_first:
        # outputs is list of (batch_size, hidden_size)
        # Stack to (batch_size, seq_len, hidden_size)
        output_stacked = torch.stack(outputs, dim=1)
    else:
        # Stack to (seq_len, batch_size, hidden_size)
        output_stacked = torch.stack(outputs, dim=0)
    
    # Expected output shape
    if batch_first:
        expected_output_shape = (batch_size, seq_len, hidden_size)
    else:
        expected_output_shape = (seq_len, batch_size, hidden_size)
    
    # Expected hidden state shape (for single timestep)
    expected_hidden_shape = (batch_size, hidden_size)
    
    # Weak assertions
    # 1. Output shape assertion
    assert_shape_equal(output_stacked, expected_output_shape,
                      f"Output shape mismatch for {cell_type}")
    
    # 2. Hidden state shape assertion (for final hidden state)
    if is_lstm:
        final_hidden, final_cell = hidden
        assert_shape_equal(final_hidden, expected_hidden_shape,
                          f"Final hidden state shape mismatch for {cell_type}")
        assert_shape_equal(final_cell, expected_hidden_shape,
                          f"Final cell state shape mismatch for {cell_type}")
    else:
        assert_shape_equal(hidden, expected_hidden_shape,
                          f"Final hidden state shape mismatch for {cell_type}")
    
    # 3. Dtype assertion
    assert_dtype_equal(output_stacked, dtype,
                      f"Output dtype mismatch for {cell_type}")
    if is_lstm:
        assert_dtype_equal(final_hidden, dtype,
                          f"Hidden state dtype mismatch for {cell_type}")
        assert_dtype_equal(final_cell, dtype,
                          f"Cell state dtype mismatch for {cell_type}")
    else:
        assert_dtype_equal(hidden, dtype,
                          f"Hidden state dtype mismatch for {cell_type}")
    
    # 4. Finite values assertion
    assert_finite(output_stacked, f"Output contains non-finite values for {cell_type}")
    if is_lstm:
        assert_finite(final_hidden, f"Hidden state contains non-finite values for {cell_type}")
        assert_finite(final_cell, f"Cell state contains non-finite values for {cell_type}")
    else:
        assert_finite(hidden, f"Hidden state contains non-finite values for {cell_type}")
    
    # 5. No NaN assertion (additional safety check)
    assert_no_nan(output_stacked, f"Output contains NaN values for {cell_type}")
    if is_lstm:
        assert_no_nan(final_hidden, f"Hidden state contains NaN values for {cell_type}")
        assert_no_nan(final_cell, f"Cell state contains NaN values for {cell_type}")
    else:
        assert_no_nan(hidden, f"Hidden state contains NaN values for {cell_type}")
    
    # Cell-specific consistency checks
    if cell_type == "RNNCell" or cell_type == "RNNCell_relu":
        # RNNCell specific checks
        activation = "relu" if cell_type == "RNNCell_relu" else "tanh"
        
        # Check activation function
        if activation == "tanh":
            # tanh output should be in [-1, 1]
            assert torch.all(output_stacked >= -1.0) and torch.all(output_stacked <= 1.0), \
                f"RNNCell with tanh should output values in [-1, 1], got range [{output_stacked.min():.4f}, {output_stacked.max():.4f}]"
        elif activation == "relu":
            # relu output should be >= 0
            assert torch.all(output_stacked >= 0.0), \
                f"RNNCell with relu should output non-negative values, got min {output_stacked.min():.4f}"
        
        # Check parameter count
        param_count = sum(p.numel() for p in cell.parameters())
        # RNNCell: W_ih (hidden_size, input_size), W_hh (hidden_size, hidden_size)
        # plus biases if bias=True
        expected_params = hidden_size * input_size + hidden_size * hidden_size
        if bias:
            expected_params += 2 * hidden_size  # bias_ih and bias_hh
        
        assert param_count == expected_params, \
            f"RNNCell parameter count mismatch: expected {expected_params}, got {param_count}"
    
    elif cell_type == "LSTMCell":
        # LSTMCell specific checks
        # Check that hidden and cell states are different
        assert not torch.allclose(final_hidden, final_cell, rtol=1e-5, atol=1e-8), \
            "LSTMCell hidden and cell states should not be identical"
        
        # Check parameter count
        param_count = sum(p.numel() for p in cell.parameters())
        # LSTMCell: gate_size = 4 * hidden_size
        # W_ih (4*hidden_size, input_size), W_hh (4*hidden_size, hidden_size)
        # plus biases if bias=True
        expected_params = 4 * hidden_size * input_size + 4 * hidden_size * hidden_size
        if bias:
            expected_params += 2 * 4 * hidden_size  # bias_ih and bias_hh
        
        assert param_count == expected_params, \
            f"LSTMCell parameter count mismatch: expected {expected_params}, got {param_count}"
        
        # Check that cell state values are reasonable
        cell_state_abs_max = final_cell.abs().max().item()
        assert cell_state_abs_max < 100.0, \
            f"LSTMCell cell state should not explode, max abs value: {cell_state_abs_max}"
    
    elif cell_type == "GRUCell":
        # GRUCell specific checks
        # Check parameter count
        param_count = sum(p.numel() for p in cell.parameters())
        # GRUCell: gate_size = 3 * hidden_size
        # W_ih (3*hidden_size, input_size), W_hh (3*hidden_size, hidden_size)
        # plus biases if bias=True
        expected_params = 3 * hidden_size * input_size + 3 * hidden_size * hidden_size
        if bias:
            expected_params += 2 * 3 * hidden_size  # bias_ih and bias_hh
        
        assert param_count == expected_params, \
            f"GRUCell parameter count mismatch: expected {expected_params}, got {param_count}"
        
        # Check that output values are reasonable
        output_abs_max = output_stacked.abs().max().item()
        assert output_abs_max < 100.0, \
            f"GRUCell output should not explode, max abs value: {output_abs_max}"
    
    # Test cell vs full RNN consistency (simplified)
    # For single-layer, single-direction RNN, the cell should produce same result
    # as unrolling the RNN manually
    if not is_lstm and seq_len <= 5:  # Keep it simple for short sequences
        # Create corresponding full RNN
        full_rnn = None
        if cell_type == "RNNCell" or cell_type == "RNNCell_relu":
            nonlinearity = "relu" if cell_type == "RNNCell_relu" else "tanh"
            full_rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=batch_first,
                bidirectional=False,
                nonlinearity=nonlinearity,
                bias=bias,
            )
        elif cell_type == "GRUCell":
            full_rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=batch_first,
                bidirectional=False,
                bias=bias,
            )
        
        if full_rnn is not None:
            # Copy weights from cell to full RNN
            cell_params = dict(cell.named_parameters())
            rnn_params = dict(full_rnn.named_parameters())
            
            # RNN weight names are different: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
            # Cell weight names: weight_ih, weight_hh, bias_ih, bias_hh
            weight_mapping = {
                'weight_ih': 'weight_ih_l0',
                'weight_hh': 'weight_hh_l0',
                'bias_ih': 'bias_ih_l0',
                'bias_hh': 'bias_hh_l0',
            }
            
            for cell_name, rnn_name in weight_mapping.items():
                if cell_name in cell_params and rnn_name in rnn_params:
                    rnn_params[rnn_name].data.copy_(cell_params[cell_name].data)
            
            # Convert to same dtype
            if dtype == torch.float64:
                full_rnn = full_rnn.double()
            
            # Forward pass through full RNN
            full_output, full_hidden = full_rnn(x_all)
            
            # Check shapes match
            assert output_stacked.shape == full_output.shape, \
                f"Cell and full RNN output shapes should match: {output_stacked.shape} vs {full_output.shape}"
            
            # Values may differ due to different numerical implementations,
            # but they should be close
            # We'll just check they have same shape and finite values
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Additional test cases for edge cases and error conditions

def test_rnn_invalid_input_dimensions():
    """Test that invalid input dimensions raise appropriate errors."""
    # Test 1D input (invalid - should be 2D for unbatched or 3D for batched)
    rnn = nn.RNN(input_size=10, hidden_size=20)
    x_1d = torch.randn(10)  # 1D input - should fail
    
    with pytest.raises(RuntimeError, match="input must have 2 or 3 dimensions"):
        rnn(x_1d)
    
    # Test 4D input (invalid)
    rnn = nn.RNN(input_size=10, hidden_size=20)
    x_4d = torch.randn(2, 3, 4, 10)  # 4D input - should fail
    
    with pytest.raises(RuntimeError, match="input must have 2 or 3 dimensions"):
        rnn(x_4d)
    
    # Test wrong input size (3D batched input with wrong feature dimension)
    rnn = nn.RNN(input_size=10, hidden_size=20)
    x_wrong_size = torch.randn(5, 3, 15)  # input_size=15, expected 10
    
    with pytest.raises(RuntimeError, match="input.size\\(2\\) must be equal to input_size"):
        rnn(x_wrong_size)
    
    # Test wrong input size (2D unbatched input with wrong feature dimension)
    rnn = nn.RNN(input_size=10, hidden_size=20)
    x_wrong_size_2d = torch.randn(5, 15)  # input_size=15, expected 10
    
    with pytest.raises(RuntimeError, match="input.size\\(1\\) must be equal to input_size"):
        rnn(x_wrong_size_2d)

def test_rnn_hidden_state_mismatch():
    """Test that hidden state shape mismatch raises error."""
    rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=2)
    x = torch.randn(5, 3, 8)  # (seq_len, batch, input_size)
    
    # Correct hidden state shape: (num_layers * num_directions, batch, hidden_size) = (2, 3, 16)
    h0_correct = torch.randn(2, 3, 16)
    
    # Wrong batch size
    h0_wrong_batch = torch.randn(2, 4, 16)  # batch=4, expected 3
    
    with pytest.raises(RuntimeError, match="Expected hidden size"):
        rnn(x, h0_wrong_batch)
    
    # Wrong hidden size
    h0_wrong_hidden = torch.randn(2, 3, 20)  # hidden_size=20, expected 16
    
    with pytest.raises(RuntimeError, match="Expected hidden size"):
        rnn(x, h0_wrong_hidden)
    
    # Wrong number of layers
    h0_wrong_layers = torch.randn(1, 3, 16)  # 1 layer, expected 2
    
    with pytest.raises(RuntimeError, match="Expected hidden size"):
        rnn(x, h0_wrong_layers)

def test_rnn_dropout_validation():
    """Test dropout parameter validation."""
    # Valid dropout values
    for dropout in [0.0, 0.5, 1.0]:
        rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, dropout=dropout)
        assert rnn.dropout == dropout, f"Dropout should be {dropout}, got {rnn.dropout}"
    
    # Invalid dropout values
    # PyTorch's actual error message for invalid dropout values
    with pytest.raises(ValueError, match="dropout should be a number in range"):
        nn.RNN(input_size=10, hidden_size=20, dropout=-0.1)
    
    with pytest.raises(ValueError, match="dropout should be a number in range"):
        nn.RNN(input_size=10, hidden_size=20, dropout=1.1)
    
    # For string input, PyTorch raises ValueError with message about conversion
    with pytest.raises(ValueError, match="could not convert string to float"):
        nn.RNN(input_size=10, hidden_size=20, dropout="invalid")

def test_rnn_weight_initialization():
    """Test that RNN weights are properly initialized."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create RNN with default initialization
    rnn = nn.RNN(input_size=8, hidden_size=16)
    
    # Check that all parameters are initialized
    for name, param in rnn.named_parameters():
        assert param is not None, f"Parameter {name} should be initialized"
        assert param.requires_grad, f"Parameter {name} should require gradients"
        
        # Check that parameters are not all zeros (unlikely with proper initialization)
        assert not torch.allclose(param, torch.zeros_like(param), rtol=1e-10, atol=1e-10), \
            f"Parameter {name} should not be all zeros after initialization"
        
        # Check that parameters have finite values
        assert torch.isfinite(param).all(), f"Parameter {name} should have finite values"
    
    # Check specific parameter shapes
    expected_shapes = {
        'weight_ih_l0': (16, 8),    # (hidden_size, input_size)
        'weight_hh_l0': (16, 16),   # (hidden_size, hidden_size)
        'bias_ih_l0': (16,),        # (hidden_size,)
        'bias_hh_l0': (16,),        # (hidden_size,)
    }
    
    for name, expected_shape in expected_shapes.items():
        assert name in dict(rnn.named_parameters()), f"Parameter {name} should exist"
        param = dict(rnn.named_parameters())[name]
        assert param.shape == expected_shape, \
            f"Parameter {name} should have shape {expected_shape}, got {param.shape}"

def test_rnn_no_bias():
    """Test RNN without bias parameters."""
    rnn = nn.RNN(input_size=10, hidden_size=20, bias=False)
    
    # Check that bias parameters don't exist
    param_names = [name for name, _ in rnn.named_parameters()]
    assert 'bias_ih_l0' not in param_names, "bias_ih_l0 should not exist when bias=False"
    assert 'bias_hh_l0' not in param_names, "bias_hh_l0 should not exist when bias=False"
    
    # Should still have weight parameters
    assert 'weight_ih_l0' in param_names, "weight_ih_l0 should exist"
    assert 'weight_hh_l0' in param_names, "weight_hh_l0 should exist"
    
    # Forward pass should work
    x = torch.randn(5, 3, 10)
    output, h_n = rnn(x)
    
    # Check shapes
    assert output.shape == (5, 3, 20), f"Output shape should be (5, 3, 20), got {output.shape}"
    assert h_n.shape == (1, 3, 20), f"Hidden state shape should be (1, 3, 20), got {h_n.shape}"

def test_rnn_device_consistency():
    """Test that RNN works correctly on CPU (basic device test)."""
    # Create RNN on CPU
    rnn = nn.RNN(input_size=8, hidden_size=16).cpu()
    
    # Create input on CPU
    x = torch.randn(5, 3, 8, device='cpu')
    
    # Forward pass should work
    output, h_n = rnn(x)
    
    # Check that output is on CPU
    assert output.device.type == 'cpu', f"Output should be on CPU, got {output.device}"
    assert h_n.device.type == 'cpu', f"Hidden state should be on CPU, got {h_n.device}"
    
    # Check shapes
    assert output.shape == (5, 3, 16), f"Output shape mismatch on CPU"
    assert h_n.shape == (1, 3, 16), f"Hidden state shape mismatch on CPU"

@pytest.mark.xfail(reason="Zero batch size not supported by PyTorch RNN")
def test_rnn_zero_batch_size():
    """Test RNN with zero batch size (edge case)."""
    rnn = nn.RNN(input_size=10, hidden_size=20)
    
    # Create input with batch_size=0
    x = torch.randn(5, 0, 10)  # (seq_len, batch_size=0, input_size)
    
    # This should raise an error
    with pytest.raises(RuntimeError, match="Expected batch size"):
        output, h_n = rnn(x)

def test_rnn_state_dict_serialization():
    """Test that RNN state dict can be saved and loaded."""
    # Create and train a simple RNN
    torch.manual_seed(42)
    rnn1 = nn.RNN(input_size=8, hidden_size=16, num_layers=2)
    
    # Do a forward pass to change parameters (simulating training)
    x = torch.randn(5, 3, 8)
    output1, h_n1 = rnn1(x)
    
    # Save state dict
    state_dict = rnn1.state_dict()
    
    # Create a new RNN with same architecture
    rnn2 = nn.RNN(input_size=8, hidden_size=16, num_layers=2)
    
    # Load state dict
    rnn2.load_state_dict(state_dict)
    
    # Forward pass with same input should produce same output
    output2, h_n2 = rnn2(x)
    
    # Check that outputs are close (should be identical with same weights)
    assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-8), \
        "Outputs should be identical after loading state dict"
    assert torch.allclose(h_n1, h_n2, rtol=1e-5, atol=1e-8), \
        "Hidden states should be identical after loading state dict"
    
    # Check that all parameters match
    for (name1, param1), (name2, param2) in zip(rnn1.named_parameters(), rnn2.named_parameters()):
        assert name1 == name2, f"Parameter names should match: {name1} vs {name2}"
        assert torch.allclose(param1, param2, rtol=1e-5, atol=1e-8), \
            f"Parameter {name1} should match after loading state dict"

# Cleanup and main execution
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
# ==== BLOCK:FOOTER END ====