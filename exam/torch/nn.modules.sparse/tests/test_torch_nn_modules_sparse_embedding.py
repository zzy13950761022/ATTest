import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.sparse import Embedding

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions
@pytest.fixture(scope="function")
def set_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    return 42

def assert_tensor_properties(tensor, expected_shape=None, expected_dtype=None, 
                           allow_nan_inf=False, name=""):
    """Helper to assert tensor properties"""
    assert isinstance(tensor, torch.Tensor), f"{name} should be a torch.Tensor"
    
    if expected_shape is not None:
        assert tensor.shape == expected_shape, \
            f"{name} shape mismatch: {tensor.shape} != {expected_shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, \
            f"{name} dtype mismatch: {tensor.dtype} != {expected_dtype}"
    
    if not allow_nan_inf:
        assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf values"
    
    return True

def create_test_input(shape, dtype=torch.long, device="cpu"):
    """Create test input tensor with valid indices"""
    if dtype not in (torch.long, torch.int):
        raise ValueError(f"Input dtype must be torch.long or torch.int, got {dtype}")
    
    # Create random indices in valid range [0, num_embeddings-1]
    # For testing, we'll use a fixed range
    tensor = torch.randint(0, 10, shape, dtype=dtype, device=device)
    return tensor
# ==== BLOCK:HEADER END ====

class TestEmbedding:
    """Test class for torch.nn.modules.sparse.Embedding"""
    
    # ==== BLOCK:CASE_01 START ====
    @pytest.mark.parametrize("num_embeddings,embedding_dim,padding_idx,max_norm,sparse,input_shape,dtype,device", [
        (10, 3, None, None, False, (5,), torch.long, "cpu"),
    ])
    def test_embedding_basic_forward(self, set_random_seed, num_embeddings, embedding_dim, 
                                    padding_idx, max_norm, sparse, input_shape, dtype, device):
        """TC-01: Embedding 基础正向传播"""
        # Create Embedding module
        embedding = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            sparse=sparse
        )
        
        # Create test input
        indices = torch.randint(0, num_embeddings, input_shape, dtype=dtype, device=device)
        
        # Forward pass
        output = embedding(indices)
        
        # Weak assertions
        # 1. Shape assertion
        expected_shape = input_shape + (embedding_dim,)
        assert output.shape == expected_shape, \
            f"Output shape mismatch: {output.shape} != {expected_shape}"
        
        # 2. Dtype assertion
        assert output.dtype == torch.float32, \
            f"Output dtype should be float32, got {output.dtype}"
        
        # 3. Finite values assertion
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
        # 4. Basic property: output should not be all zeros
        assert not torch.allclose(output, torch.zeros_like(output)), \
            "Output should not be all zeros"
        
        # 5. Compare with functional embedding as oracle
        # Note: This is a weak comparison, just checking shape and dtype match
        func_output = F.embedding(
            indices, 
            embedding.weight, 
            padding_idx=padding_idx,
            max_norm=max_norm
        )
        
        assert func_output.shape == output.shape, \
            f"Functional output shape mismatch: {func_output.shape} != {output.shape}"
        assert func_output.dtype == output.dtype, \
            f"Functional output dtype mismatch: {func_output.dtype} != {output.dtype}"
        
        # 6. Verify weight access
        # The embedding layer should have a weight parameter
        assert hasattr(embedding, 'weight'), "Embedding should have weight attribute"
        assert embedding.weight.shape == (num_embeddings, embedding_dim), \
            f"Weight shape mismatch: {embedding.weight.shape} != ({num_embeddings}, {embedding_dim})"
        
        # 7. Verify that different indices produce different outputs
        # (unless they map to the same embedding vector)
        if input_shape[0] > 1:
            # Check that at least some outputs are different
            unique_rows = torch.unique(output.view(-1, embedding_dim), dim=0)
            assert unique_rows.shape[0] > 1, \
                "Different indices should produce different embedding vectors"
    # ==== BLOCK:CASE_01 END ====
    
    # ==== BLOCK:CASE_02 START ====
    @pytest.mark.parametrize("num_embeddings,embedding_dim,padding_idx,max_norm,sparse,input_shape,dtype,device", [
        (10, 3, 0, None, False, (5,), torch.long, "cpu"),
        # Parameter extension: negative padding_idx (Medium priority)
        (10, 3, -1, None, False, (5,), torch.long, "cpu"),
    ])
    def test_embedding_padding_idx(self, set_random_seed, num_embeddings, embedding_dim, 
                                  padding_idx, max_norm, sparse, input_shape, dtype, device):
        """TC-02: Embedding padding_idx 处理"""
        # Create Embedding module with padding_idx
        embedding = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            sparse=sparse
        )
        
        # Create test input with some indices equal to padding_idx
        # Note: For negative padding_idx, we need to use the corresponding positive index
        # padding_idx=-1 means the last embedding (index num_embeddings-1)
        if padding_idx is not None and padding_idx < 0:
            # Convert negative padding_idx to positive for test indices
            actual_padding_idx = num_embeddings + padding_idx
        else:
            actual_padding_idx = padding_idx
        
        indices = torch.randint(0, num_embeddings, input_shape, dtype=dtype, device=device)
        # Set first element to padding_idx (use actual positive index)
        if actual_padding_idx is not None:
            indices[0] = actual_padding_idx
        
        # Forward pass
        output = embedding(indices)
        
        # Weak assertions
        # 1. Shape assertion
        expected_shape = input_shape + (embedding_dim,)
        assert output.shape == expected_shape, \
            f"Output shape mismatch: {output.shape} != {expected_shape}"
        
        # 2. Dtype assertion
        assert output.dtype == torch.float32, \
            f"Output dtype should be float32, got {output.dtype}"
        
        # 3. Finite values assertion
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
        # 4. Padding zero assertion: positions with padding_idx should output zeros
        if actual_padding_idx is not None:
            padding_mask = (indices == actual_padding_idx)
            padding_outputs = output[padding_mask]
            assert torch.allclose(padding_outputs, torch.zeros_like(padding_outputs)), \
                "Padding indices should produce zero output"
        
        # 5. Non-padding positions should have non-zero output
        if actual_padding_idx is not None:
            non_padding_mask = (indices != actual_padding_idx)
        else:
            non_padding_mask = torch.ones_like(indices, dtype=torch.bool)
        if non_padding_mask.any():
            non_padding_outputs = output[non_padding_mask]
            assert not torch.allclose(non_padding_outputs, torch.zeros_like(non_padding_outputs)), \
                "Non-padding indices should produce non-zero output"
        
        # 6. Gradient isolation check (weak)
        # Create a simple loss and compute gradients
        output.sum().backward()
        
        # Check that weight gradient for padding_idx is zero
        if embedding.weight.grad is not None and actual_padding_idx is not None:
            padding_grad = embedding.weight.grad[actual_padding_idx]
            assert torch.allclose(padding_grad, torch.zeros_like(padding_grad), rtol=1e-5), \
                "Gradient for padding_idx should be zero"
            
            # Check that other gradients are non-zero (if there are non-padding indices)
            if non_padding_mask.any():
                non_padding_indices = indices[non_padding_mask].unique()
                for idx in non_padding_indices:
                    if idx != actual_padding_idx:
                        grad = embedding.weight.grad[idx]
                        assert not torch.allclose(grad, torch.zeros_like(grad), rtol=1e-5), \
                            f"Gradient for non-padding index {idx} should be non-zero"
        
        # 7. Compare with functional embedding
        func_output = F.embedding(
            indices, 
            embedding.weight, 
            padding_idx=padding_idx,
            max_norm=max_norm
        )
        
        assert func_output.shape == output.shape, \
            f"Functional output shape mismatch: {func_output.shape} != {output.shape}"
        assert func_output.dtype == output.dtype, \
            f"Functional output dtype mismatch: {func_output.dtype} != {output.dtype}"
        
        # For padding positions, both should be zero
        if actual_padding_idx is not None:
            func_padding_outputs = func_output[padding_mask]
            assert torch.allclose(func_padding_outputs, torch.zeros_like(func_padding_outputs)), \
                "Functional embedding should also produce zero for padding indices"
    # ==== BLOCK:CASE_02 END ====
    
    # ==== BLOCK:CASE_05 START ====
    @pytest.mark.parametrize("num_embeddings,embedding_dim,padding_idx,max_norm,norm_type,sparse,input_shape,dtype,device", [
        (10, 3, None, 1.0, 2.0, False, (5,), torch.long, "cpu"),
    ])
    def test_embedding_max_norm_constraint(self, set_random_seed, num_embeddings, embedding_dim, 
                                          padding_idx, max_norm, norm_type, sparse, input_shape, dtype, device):
        """TC-05: Embedding max_norm 约束"""
        # Create Embedding module with max_norm constraint
        embedding = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            sparse=sparse
        )
        
        # Create test input
        indices = torch.randint(0, num_embeddings, input_shape, dtype=dtype, device=device)
        
        # Forward pass
        output = embedding(indices)
        
        # Weak assertions
        # 1. Shape assertion
        expected_shape = input_shape + (embedding_dim,)
        assert output.shape == expected_shape, \
            f"Output shape mismatch: {output.shape} != {expected_shape}"
        
        # 2. Dtype assertion
        assert output.dtype == torch.float32, \
            f"Output dtype should be float32, got {output.dtype}"
        
        # 3. Norm constraint assertion
        # Check that the weight matrix has been normalized (in-place modification)
        # Get the weight after forward pass (it may have been modified)
        weight_after = embedding.weight.detach().clone()
        
        # Compute norms of each embedding vector
        weight_norms = torch.norm(weight_after, p=norm_type, dim=1)
        
        # All norms should be <= max_norm (with some tolerance)
        # Note: max_norm constraint is applied in-place during forward pass
        assert (weight_norms <= max_norm * 1.01).all(), \
            f"Weight norms exceed max_norm: max {weight_norms.max().item()} > {max_norm}"
        
        # 4. Weight modification assertion
        # The weight should have been modified in-place (for vectors exceeding max_norm)
        # We can't easily check this without knowing initial weights,
        # but we can verify that at least some vectors have norm <= max_norm
        
        # 5. Finite values assertion
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
        # 6. Compare with functional embedding
        func_output = F.embedding(
            indices, 
            embedding.weight,  # Use the potentially modified weight
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type
        )
        
        assert func_output.shape == output.shape, \
            f"Functional output shape mismatch: {func_output.shape} != {output.shape}"
        assert func_output.dtype == output.dtype, \
            f"Functional output dtype mismatch: {func_output.dtype} != {output.dtype}"
        
        # 7. Basic property: output should not be all zeros
        assert not torch.allclose(output, torch.zeros_like(output)), \
            "Output should not be all zeros"
    # ==== BLOCK:CASE_05 END ====
    
    # ==== BLOCK:CASE_06 START ====
    @pytest.mark.parametrize("num_embeddings,embedding_dim,padding_idx,max_norm,sparse,input_shape,dtype,device", [
        (10, 3, None, None, True, (5,), torch.long, "cpu"),
    ])
    def test_embedding_sparse_gradient_mode(self, set_random_seed, num_embeddings, embedding_dim, 
                                           padding_idx, max_norm, sparse, input_shape, dtype, device):
        """TC-06: Embedding 稀疏梯度模式"""
        # Test placeholder - deferred
        pass
    # ==== BLOCK:CASE_06 END ====

# ==== BLOCK:FOOTER START ====
# Additional test cases and edge cases

def test_embedding_invalid_parameters():
    """Test invalid parameter combinations"""
    # Test 1: num_embeddings <= 0 should raise RuntimeError (not ValueError)
    # Embedding constructor doesn't validate num_embeddings directly,
    # but creating weight tensor with invalid dimensions will fail
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        Embedding(num_embeddings=0, embedding_dim=3)
    # Check that some error is raised
    assert exc_info.value is not None
    
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        Embedding(num_embeddings=-1, embedding_dim=3)
    assert exc_info.value is not None
    
    # Test 2: embedding_dim <= 0 should raise RuntimeError (not ValueError)
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        Embedding(num_embeddings=10, embedding_dim=0)
    assert exc_info.value is not None
    
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        Embedding(num_embeddings=10, embedding_dim=-1)
    assert exc_info.value is not None
    
    # Test 3: padding_idx out of range should raise AssertionError (not ValueError)
    # Embedding uses assert statements for padding_idx validation
    with pytest.raises(AssertionError, match="padding_idx must be within"):
        Embedding(num_embeddings=10, embedding_dim=3, padding_idx=10)
    
    with pytest.raises(AssertionError, match="padding_idx must be within"):
        Embedding(num_embeddings=10, embedding_dim=3, padding_idx=-11)
    
    # Test 4: max_norm <= 0 should raise ValueError
    with pytest.raises(ValueError, match="max_norm must be positive"):
        Embedding(num_embeddings=10, embedding_dim=3, max_norm=0)
    
    with pytest.raises(ValueError, match="max_norm must be positive"):
        Embedding(num_embeddings=10, embedding_dim=3, max_norm=-1.0)
    
    # Test 5: Invalid input dtype should raise TypeError
    embedding = Embedding(num_embeddings=10, embedding_dim=3)
    with pytest.raises(RuntimeError, match="Expected tensor for argument #1 'indices'"):
        embedding(torch.tensor([1.0, 2.0, 3.0]))  # Float tensor instead of Long

def test_embedding_edge_cases():
    """Test edge cases for Embedding"""
    torch.manual_seed(42)
    
    # Test 1: Single element input
    embedding = Embedding(num_embeddings=10, embedding_dim=3)
    indices = torch.tensor([5], dtype=torch.long)
    output = embedding(indices)
    assert output.shape == (1, 3)
    assert torch.isfinite(output).all()
    
    # Test 2: 2D input (batch processing)
    embedding = Embedding(num_embeddings=10, embedding_dim=3)
    indices = torch.randint(0, 10, (2, 4), dtype=torch.long)
    output = embedding(indices)
    assert output.shape == (2, 4, 3)
    
    # Test 3: Large dimensions
    embedding = Embedding(num_embeddings=1000, embedding_dim=256)
    indices = torch.randint(0, 1000, (32, 10), dtype=torch.long)  # batch of 32, seq length 10
    output = embedding(indices)
    assert output.shape == (32, 10, 256)
    assert torch.isfinite(output).all()
    
    # Test 4: Negative padding_idx
    # Note: negative padding_idx is converted to positive index internally
    # padding_idx=-1 means the last embedding (index 9)
    embedding = Embedding(num_embeddings=10, embedding_dim=3, padding_idx=-1)
    indices = torch.tensor([1, 2, -1, 4], dtype=torch.long)
    # Note: index -1 is not valid for embedding lookup, it will cause IndexError
    # We need to use valid indices for testing
    indices = torch.tensor([1, 2, 9, 4], dtype=torch.long)  # Use 9 instead of -1
    output = embedding(indices)
    assert output.shape == (4, 3)
    # Position with index 9 (which is padding_idx after conversion) should be zero
    assert torch.allclose(output[2], torch.zeros(3))
    
    # Test 5: Empty input
    embedding = Embedding(num_embeddings=10, embedding_dim=3)
    indices = torch.empty(0, dtype=torch.long)
    output = embedding(indices)
    assert output.shape == (0, 3)
    
    # Test 6: All indices are padding_idx
    embedding = Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)
    indices = torch.zeros(5, dtype=torch.long)
    output = embedding(indices)
    assert output.shape == (5, 3)
    assert torch.allclose(output, torch.zeros_like(output))
    
    # Test 7: Index at boundary (0 and num_embeddings-1)
    embedding = Embedding(num_embeddings=10, embedding_dim=3)
    indices = torch.tensor([0, 9], dtype=torch.long)  # First and last indices
    output = embedding(indices)
    assert output.shape == (2, 3)
    # Both should produce valid embeddings
    assert torch.isfinite(output).all()
    assert not torch.allclose(output[0], torch.zeros(3))
    assert not torch.allclose(output[1], torch.zeros(3))
# ==== BLOCK:FOOTER END ====