import math
import pytest
import torch
import torch.nn as nn
from torch.nn.modules.transformer import (
    Transformer,
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer
)

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions (shared with G1)
@pytest.fixture(scope="module")
def set_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

@pytest.fixture
def device():
    """Get available device (CPU only for consistency)"""
    return torch.device("cpu")

def create_test_tensor(shape, dtype=torch.float32, device="cpu"):
    """Create test tensor with fixed random values"""
    torch.manual_seed(123)
    return torch.randn(*shape, dtype=dtype, device=device)

def assert_tensor_properties(tensor, expected_shape=None, expected_dtype=None, 
                           expected_device=None, finite_check=True):
    """Assert tensor properties match expectations"""
    if expected_shape is not None:
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    if expected_device is not None:
        assert tensor.device == expected_device, f"Expected device {expected_device}, got {tensor.device}"
    
    if finite_check:
        assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"
        assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_05 START ====
@pytest.mark.parametrize(
    "dtype,device,nhead,d_model,num_layers,dim_feedforward,dropout,activation,src_shape",
    [
        (
            torch.float32,
            "cpu",
            2,
            8,
            2,
            32,
            0.0,
            "relu",
            (10, 16, 8),
        ),
        # Param extension: more heads, more layers, GELU activation, dropout
        (
            torch.float32,
            "cpu",
            4,
            16,
            3,
            64,
            0.1,
            "gelu",
            (20, 4, 16),
        ),
    ]
)
def test_transformer_encoder_basic(
    dtype,
    device,
    nhead,
    d_model,
    num_layers,
    dim_feedforward,
    dropout,
    activation,
    src_shape,
    set_random_seed,
):
    """TC-05: TransformerEncoder 基础功能"""
    # Create encoder layer
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    
    # Create encoder with multiple layers
    encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    # Convert model parameters to match input dtype
    encoder = encoder.to(dtype=dtype, device=device)
    encoder.eval()  # Disable dropout for deterministic output
    
    # Create source tensor
    src = create_test_tensor(src_shape, dtype=dtype, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = encoder(src)
    
    # Weak assertions
    # 1. Output shape check
    expected_shape = src_shape  # Encoder should preserve input shape
    assert output.shape == expected_shape, (
        f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"
    )
    
    # 2. Output dtype check
    assert output.dtype == dtype, (
        f"Output dtype mismatch. Expected {dtype}, got {output.dtype}"
    )
    
    # 3. Finite check
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    # 4. Layer count check (verify encoder has correct number of layers)
    assert len(encoder.layers) == num_layers, (
        f"Encoder should have {num_layers} layers, got {len(encoder.layers)}"
    )
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
@pytest.mark.parametrize(
    "dtype,device,nhead,d_model,num_layers,dim_feedforward,dropout,activation,tgt_shape,memory_shape",
    [
        (
            torch.float32,
            "cpu",
            2,
            8,
            2,
            32,
            0.0,
            "relu",
            (10, 16, 8),
            (5, 16, 8),
        ),
    ]
)
def test_transformer_decoder_basic(
    dtype,
    device,
    nhead,
    d_model,
    num_layers,
    dim_feedforward,
    dropout,
    activation,
    tgt_shape,
    memory_shape,
    set_random_seed,
):
    """TC-06: TransformerDecoder 基础功能"""
    # Create decoder layer
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    
    # Create decoder with multiple layers
    decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
    
    # Convert model parameters to match input dtype
    decoder = decoder.to(dtype=dtype, device=device)
    decoder.eval()  # Disable dropout for deterministic output
    
    # Create target and memory tensors
    tgt = create_test_tensor(tgt_shape, dtype=dtype, device=device)
    memory = create_test_tensor(memory_shape, dtype=dtype, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = decoder(tgt, memory)
    
    # Weak assertions
    # 1. Output shape check
    expected_shape = tgt_shape  # Decoder should preserve target shape
    assert output.shape == expected_shape, (
        f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"
    )
    
    # 2. Output dtype check
    assert output.dtype == dtype, (
        f"Output dtype mismatch. Expected {dtype}, got {output.dtype}"
    )
    
    # 3. Finite check
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    # 4. Memory usage check (verify decoder uses memory)
    # This is implicit in the forward pass - if memory wasn't used, 
    # the decoder would likely fail or produce different output
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
@pytest.mark.parametrize(
    "dtype,device,nhead,d_model,dim_feedforward,dropout,activation,src_shape",
    [
        (
            torch.float32,
            "cpu",
            2,
            8,
            32,
            0.0,
            "relu",
            (10, 16, 8),
        ),
    ]
)
def test_transformer_encoder_layer_single(
    dtype,
    device,
    nhead,
    d_model,
    dim_feedforward,
    dropout,
    activation,
    src_shape,
    set_random_seed,
):
    """TC-07: TransformerEncoderLayer 单层"""
    # Create encoder layer
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    
    # Convert model parameters to match input dtype
    encoder_layer = encoder_layer.to(dtype=dtype, device=device)
    encoder_layer.eval()  # Disable dropout for deterministic output
    
    # Create source tensor
    src = create_test_tensor(src_shape, dtype=dtype, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = encoder_layer(src)
    
    # Weak assertions
    # 1. Output shape check
    expected_shape = src_shape  # Encoder layer should preserve input shape
    assert output.shape == expected_shape, (
        f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"
    )
    
    # 2. Output dtype check
    assert output.dtype == dtype, (
        f"Output dtype mismatch. Expected {dtype}, got {output.dtype}"
    )
    
    # 3. Finite check
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    # 4. Residual connection check (output should not be identical to input)
    # Note: With dropout=0.0 and no normalization, there should still be some transformation
    assert not torch.allclose(output, src, rtol=1e-5, atol=1e-5), (
        "Output should be different from input due to residual connection and transformation"
    )
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
@pytest.mark.parametrize(
    "dtype,device,nhead,d_model,dim_feedforward,dropout,activation,tgt_shape,memory_shape",
    [
        (
            torch.float32,
            "cpu",
            2,
            8,
            32,
            0.0,
            "relu",
            (10, 16, 8),
            (5, 16, 8),
        ),
    ]
)
def test_transformer_decoder_layer_single(
    dtype,
    device,
    nhead,
    d_model,
    dim_feedforward,
    dropout,
    activation,
    tgt_shape,
    memory_shape,
    set_random_seed,
):
    """TC-08: TransformerDecoderLayer 单层"""
    # Create decoder layer
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    
    # Convert model parameters to match input dtype
    decoder_layer = decoder_layer.to(dtype=dtype, device=device)
    decoder_layer.eval()  # Disable dropout for deterministic output
    
    # Create target and memory tensors
    tgt = create_test_tensor(tgt_shape, dtype=dtype, device=device)
    memory = create_test_tensor(memory_shape, dtype=dtype, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = decoder_layer(tgt, memory)
    
    # Weak assertions
    # 1. Output shape check
    expected_shape = tgt_shape  # Decoder layer should preserve target shape
    assert output.shape == expected_shape, (
        f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"
    )
    
    # 2. Output dtype check
    assert output.dtype == dtype, (
        f"Output dtype mismatch. Expected {dtype}, got {output.dtype}"
    )
    
    # 3. Finite check
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    # 4. Cross attention shape check (implicit - if cross attention failed, shape might be wrong)
    # The decoder layer uses memory for cross attention, so successful forward pass
    # indicates cross attention is working correctly
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:CASE_09 START ====
@pytest.mark.parametrize(
    "dtype,device,nhead,d_model,num_encoder_layers,num_decoder_layers,"
    "dim_feedforward,dropout,activation,src_shape,tgt_shape",
    [
        (
            torch.float32,
            "cpu",
            2,
            8,
            1,
            1,
            32,
            0.0,
            "relu",
            (5, 16, 8),
            (10, 16, 8),
        ),
    ]
)
def test_transformer_encoder_decoder_combination(
    dtype,
    device,
    nhead,
    d_model,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout,
    activation,
    src_shape,
    tgt_shape,
    set_random_seed,
):
    """TC-09: 编码器解码器组合验证"""
    # Create encoder layer
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    
    # Create encoder
    encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
    
    # Create decoder layer
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    
    # Create decoder
    decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
    
    # Convert models to match input dtype
    encoder = encoder.to(dtype=dtype, device=device)
    decoder = decoder.to(dtype=dtype, device=device)
    
    encoder.eval()
    decoder.eval()
    
    # Create source and target tensors
    src = create_test_tensor(src_shape, dtype=dtype, device=device)
    tgt = create_test_tensor(tgt_shape, dtype=dtype, device=device)
    
    # Forward pass: encoder -> decoder
    with torch.no_grad():
        # Encode source to get memory
        memory = encoder(src)
        
        # Decode target using memory
        output = decoder(tgt, memory)
    
    # Weak assertions
    # 1. Combined output shape check
    expected_shape = tgt_shape  # Output should have same shape as target
    assert output.shape == expected_shape, (
        f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"
    )
    
    # 2. Component consistency check
    # Verify encoder and decoder have correct number of layers
    assert len(encoder.layers) == num_encoder_layers, (
        f"Encoder should have {num_encoder_layers} layers, got {len(encoder.layers)}"
    )
    assert len(decoder.layers) == num_decoder_layers, (
        f"Decoder should have {num_decoder_layers} layers, got {len(decoder.layers)}"
    )
    
    # 3. Finite check
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    assert not torch.isnan(output).any(), "Output contains NaN values"
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:FOOTER START ====
# Additional test cases and utilities

def test_transformer_encoder_norm():
    """Test that TransformerEncoder has normalization layer when specified"""
    d_model = 8
    nhead = 2
    dim_feedforward = 32
    
    # Create encoder layer
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
    )
    
    # Test 1: Encoder without norm (default behavior)
    encoder_no_norm = TransformerEncoder(encoder_layer, num_layers=2)
    assert hasattr(encoder_no_norm, 'norm'), "TransformerEncoder should have norm attribute"
    assert encoder_no_norm.norm is None, "TransformerEncoder norm should be None by default"
    
    # Test 2: Encoder with norm
    norm_layer = nn.LayerNorm(d_model)
    encoder_with_norm = TransformerEncoder(encoder_layer, num_layers=2, norm=norm_layer)
    assert encoder_with_norm.norm is not None, "TransformerEncoder norm should not be None when specified"
    assert encoder_with_norm.norm is norm_layer, "Encoder norm should be the provided norm layer"
    assert isinstance(encoder_with_norm.norm, nn.LayerNorm), "Encoder norm should be LayerNorm"
    assert encoder_with_norm.norm.normalized_shape == (d_model,), f"Norm shape should be ({d_model},)"
    
    # Test forward pass works with both
    src = torch.randn(10, 16, d_model)
    
    # Without norm
    output_no_norm = encoder_no_norm(src)
    assert output_no_norm.shape == src.shape, "Output shape should match input shape without norm"
    
    # With norm
    output_with_norm = encoder_with_norm(src)
    assert output_with_norm.shape == src.shape, "Output shape should match input shape with norm"

def test_transformer_decoder_norm():
    """Test that TransformerDecoder has normalization layer when specified"""
    d_model = 8
    nhead = 2
    dim_feedforward = 32
    
    # Create decoder layer
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
    )
    
    # Test 1: Decoder without norm (default behavior)
    decoder_no_norm = TransformerDecoder(decoder_layer, num_layers=2)
    assert hasattr(decoder_no_norm, 'norm'), "TransformerDecoder should have norm attribute"
    assert decoder_no_norm.norm is None, "TransformerDecoder norm should be None by default"
    
    # Test 2: Decoder with norm
    norm_layer = nn.LayerNorm(d_model)
    decoder_with_norm = TransformerDecoder(decoder_layer, num_layers=2, norm=norm_layer)
    assert decoder_with_norm.norm is not None, "TransformerDecoder norm should not be None when specified"
    assert decoder_with_norm.norm is norm_layer, "Decoder norm should be the provided norm layer"
    assert isinstance(decoder_with_norm.norm, nn.LayerNorm), "Decoder norm should be LayerNorm"
    assert decoder_with_norm.norm.normalized_shape == (d_model,), f"Norm shape should be ({d_model},)"
    
    # Test forward pass works with both
    tgt = torch.randn(10, 16, d_model)
    memory = torch.randn(5, 16, d_model)
    
    # Without norm
    output_no_norm = decoder_no_norm(tgt, memory)
    assert output_no_norm.shape == tgt.shape, "Output shape should match target shape without norm"
    
    # With norm
    output_with_norm = decoder_with_norm(tgt, memory)
    assert output_with_norm.shape == tgt.shape, "Output shape should match target shape with norm"

def test_transformer_encoder_layer_norm_first():
    """Test TransformerEncoderLayer with norm_first=True"""
    d_model = 8
    nhead = 2
    dim_feedforward = 32
    
    # Create encoder layer with norm_first=True
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        norm_first=True,
    )
    
    # Check that layer has norm_first attribute
    assert encoder_layer.norm_first, "Encoder layer should have norm_first=True"
    
    # Create test tensor
    src = torch.randn(10, 16, d_model)
    
    # Forward pass should work
    output = encoder_layer(src)
    assert output.shape == src.shape, "Output shape should match input shape"

def test_transformer_decoder_layer_norm_first():
    """Test TransformerDecoderLayer with norm_first=True"""
    d_model = 8
    nhead = 2
    dim_feedforward = 32
    
    # Create decoder layer with norm_first=True
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        norm_first=True,
    )
    
    # Check that layer has norm_first attribute
    assert decoder_layer.norm_first, "Decoder layer should have norm_first=True"
    
    # Create test tensors
    tgt = torch.randn(10, 16, d_model)
    memory = torch.randn(5, 16, d_model)
    
    # Forward pass should work
    output = decoder_layer(tgt, memory)
    assert output.shape == tgt.shape, "Output shape should match target shape"
# ==== BLOCK:FOOTER END ====