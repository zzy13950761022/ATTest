"""
Test cases for torch.nn.modules.distance.PairwiseDistance
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== BLOCK:HEADER START ====
# Test class for PairwiseDistance
class TestPairwiseDistance:
    """Test cases for PairwiseDistance module"""
    
    def setup_method(self):
        """Setup method for each test"""
        torch.manual_seed(42)
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
    @pytest.mark.parametrize("test_params", [
        # Original test case from test_plan.json (High priority)
        {
            "dtype": torch.float32,
            "device": "cpu",
            "shape": (3, 4),
            "p": 2.0,
            "eps": 1e-6,
            "keepdim": False,
            "name": "default_params"
        },
        # Parameter extension 1: float64 with larger shape (Medium priority)
        {
            "dtype": torch.float64,
            "device": "cpu",
            "shape": (5, 6),
            "p": 2.0,
            "eps": 1e-6,
            "keepdim": False,
            "name": "float64_large_shape"
        },
        # Parameter extension 2: CUDA device (Low priority) - only if available
        {
            "dtype": torch.float32,
            "device": "cuda",
            "shape": (3, 4),
            "p": 2.0,
            "eps": 1e-6,
            "keepdim": False,
            "name": "cuda_device",
            "skip_if_no_cuda": True
        }
    ])
    def test_pairwise_distance_default_params(self, test_params):
        """CASE_01: PairwiseDistance默认参数欧氏距离（包含参数扩展）"""
        # Skip CUDA test if CUDA is not available
        if test_params.get("skip_if_no_cuda", False) and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Extract parameters
        dtype = test_params["dtype"]
        device = test_params["device"]
        shape = test_params["shape"]
        p = test_params["p"]
        eps = test_params["eps"]
        keepdim = test_params["keepdim"]
        
        # Create test inputs
        x1 = torch.randn(shape, dtype=dtype, device=device)
        x2 = torch.randn(shape, dtype=dtype, device=device)
        
        # Create PairwiseDistance module with default parameters
        pairwise_dist = nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
        
        # Forward pass
        result = pairwise_dist(x1, x2)
        
        # Oracle: torch.nn.functional.pairwise_distance
        expected = F.pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)
        
        # Weak assertions
        # 1. Shape assertion
        assert result.shape == expected.shape, \
            f"Shape mismatch: got {result.shape}, expected {expected.shape}"
        
        # 2. Dtype assertion
        assert result.dtype == expected.dtype, \
            f"Dtype mismatch: got {result.dtype}, expected {expected.dtype}"
        
        # 3. Finite values assertion
        assert torch.all(torch.isfinite(result)), \
            "Result contains non-finite values"
        
        # 4. Basic property: result should be non-negative for p >= 0
        # For p=2.0 (Euclidean distance), result should be non-negative
        if p >= 0:
            assert torch.all(result >= -eps), \
                f"Distance values should be non-negative for p={p}, got min={result.min().item()}"
        
        # 5. Approximate equality with oracle (weak assertion)
        # Use appropriate tolerance based on dtype
        if dtype == torch.float64:
            rtol = 1e-10
            atol = 1e-12
        else:
            rtol = 1e-5
            atol = 1e-6
            
        assert torch.allclose(result, expected, rtol=rtol, atol=atol), \
            f"Result doesn't match oracle. Max diff: {(result - expected).abs().max().item()}"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
    @pytest.mark.parametrize("test_params", [
        {
            "dtype": torch.float32,
            "device": "cpu",
            "shape": (2, 3),
            "p": 1.0,
            "eps": 1e-8,
            "keepdim": True
        },
        {
            "dtype": torch.float32,
            "device": "cpu",
            "shape": (2, 3),
            "p": 2.0,
            "eps": 1e-4,
            "keepdim": False
        }
    ])
    def test_pairwise_distance_param_boundaries(self, test_params):
        """CASE_02: PairwiseDistance参数边界值测试"""
        # Extract parameters
        dtype = test_params["dtype"]
        device = test_params["device"]
        shape = test_params["shape"]
        p = test_params["p"]
        eps = test_params["eps"]
        keepdim = test_params["keepdim"]
        
        # Create test inputs
        x1 = torch.randn(shape, dtype=dtype, device=device)
        x2 = torch.randn(shape, dtype=dtype, device=device)
        
        # Create PairwiseDistance module with specified parameters
        pairwise_dist = nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
        
        # Forward pass
        result = pairwise_dist(x1, x2)
        
        # Oracle: torch.nn.functional.pairwise_distance
        expected = F.pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)
        
        # Weak assertions
        # 1. Shape assertion
        assert result.shape == expected.shape, \
            f"Shape mismatch: got {result.shape}, expected {expected.shape}"
        
        # 2. Dtype assertion
        assert result.dtype == expected.dtype, \
            f"Dtype mismatch: got {result.dtype}, expected {expected.dtype}"
        
        # 3. Finite values assertion
        assert torch.all(torch.isfinite(result)), \
            "Result contains non-finite values"
        
        # 4. Basic property: result should be non-negative for p >= 0
        if p >= 0:
            assert torch.all(result >= -eps), \
                f"Distance values should be non-negative for p={p}, got min={result.min().item()}"
        
        # 5. Approximate equality with oracle (weak assertion)
        rtol = 1e-5
        atol = 1e-6
        assert torch.allclose(result, expected, rtol=rtol, atol=atol), \
            f"Result doesn't match oracle. Max diff: {(result - expected).abs().max().item()}"
        
        # 6. Keepdim effect check (basic)
        if keepdim:
            # When keepdim=True, result should have an extra dimension of size 1
            assert result.dim() == x1.dim(), \
                f"With keepdim=True, result dim should be {x1.dim()}, got {result.dim()}"
            # Last dimension should be size 1
            assert result.shape[-1] == 1, \
                f"With keepdim=True, last dimension should be 1, got {result.shape[-1]}"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_05 START ====
    def test_pairwise_distance_invalid_inputs(self):
        """CASE_05: PairwiseDistance异常输入处理"""
        # Test parameters from test_plan.json
        dtype = torch.float32
        device = "cpu"
        shape = (3, 4)
        p = 2.0
        eps = 1e-6
        keepdim = False
        
        # Create PairwiseDistance module
        pairwise_dist = nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
        
        # Test 1: Shape mismatch - different shapes
        x1 = torch.randn(shape, dtype=dtype, device=device)
        x2 = torch.randn((3, 5), dtype=dtype, device=device)  # Different shape
        
        # Weak assertion: exception should be raised
        with pytest.raises(RuntimeError) as exc_info:
            pairwise_dist(x1, x2)
        
        # Weak assertion: check exception type
        assert "RuntimeError" in str(type(exc_info.value)), \
            f"Expected RuntimeError, got {type(exc_info.value)}"
        
        # Check that error message contains shape mismatch information
        error_msg = str(exc_info.value).lower()
        # Common error messages for shape mismatch
        shape_keywords = ["shape", "size", "dimension", "must match"]
        has_shape_error = any(keyword in error_msg for keyword in shape_keywords)
        assert has_shape_error, \
            f"Error message should mention shape mismatch. Got: {error_msg}"
        
        # Test 2: Different number of dimensions
        x1_2d = torch.randn(shape, dtype=dtype, device=device)
        x2_1d = torch.randn((4,), dtype=dtype, device=device)  # 1D vs 2D
        
        with pytest.raises(RuntimeError) as exc_info2:
            pairwise_dist(x1_2d, x2_1d)
        
        # Test 3: Invalid dtype (not a tensor)
        x1_valid = torch.randn(shape, dtype=dtype, device=device)
        
        with pytest.raises(TypeError) as exc_info3:
            # Try to pass a list instead of tensor
            pairwise_dist(x1_valid, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        
        # Test 4: Empty tensors
        x1_empty = torch.randn((0, 4), dtype=dtype, device=device)
        x2_empty = torch.randn((0, 4), dtype=dtype, device=device)
        
        # Empty tensors should work (edge case)
        result_empty = pairwise_dist(x1_empty, x2_empty)
        assert result_empty.shape == (0,), \
            f"Empty tensor result shape should be (0,), got {result_empty.shape}"
        
        # Test 5: Scalar inputs (0-dimensional)
        x1_scalar = torch.tensor(1.0, dtype=dtype, device=device)
        x2_scalar = torch.tensor(2.0, dtype=dtype, device=device)
        
        # Scalar inputs should work
        result_scalar = pairwise_dist(x1_scalar, x2_scalar)
        assert result_scalar.shape == (), \
            f"Scalar result shape should be (), got {result_scalar.shape}"
        assert torch.allclose(result_scalar, torch.tensor(1.0)), \
            f"Scalar distance should be 1.0, got {result_scalar.item()}"
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
    def test_pairwise_distance_negative_p(self):
        """CASE_06: PairwiseDistance负p值测试"""
        # Test parameters from test_plan.json
        dtype = torch.float32
        device = "cpu"
        shape = (2, 3)
        p = -1.0
        eps = 1e-6
        keepdim = False
        
        # Create test inputs
        x1 = torch.randn(shape, dtype=dtype, device=device)
        x2 = torch.randn(shape, dtype=dtype, device=device)
        
        # Create PairwiseDistance module with negative p
        pairwise_dist = nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
        
        # Forward pass
        result = pairwise_dist(x1, x2)
        
        # Oracle: torch.nn.functional.pairwise_distance
        expected = F.pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)
        
        # Weak assertions
        # 1. Shape assertion
        assert result.shape == expected.shape, \
            f"Shape mismatch: got {result.shape}, expected {expected.shape}"
        
        # 2. Dtype assertion
        assert result.dtype == expected.dtype, \
            f"Dtype mismatch: got {result.dtype}, expected {expected.dtype}"
        
        # 3. Finite values assertion
        assert torch.all(torch.isfinite(result)), \
            "Result contains non-finite values"
        
        # 4. Approximate equality with oracle (weak assertion)
        rtol = 1e-5
        atol = 1e-6
        assert torch.allclose(result, expected, rtol=rtol, atol=atol), \
            f"Result doesn't match oracle. Max diff: {(result - expected).abs().max().item()}"
        
        # Additional tests for negative p behavior
        
        # Test with different negative p values
        for neg_p in [-0.5, -1.0, -2.0, -3.0]:
            pairwise_dist_neg = nn.PairwiseDistance(p=neg_p, eps=eps, keepdim=keepdim)
            result_neg = pairwise_dist_neg(x1, x2)
            expected_neg = F.pairwise_distance(x1, x2, p=neg_p, eps=eps, keepdim=keepdim)
            
            # Check finite values
            assert torch.all(torch.isfinite(result_neg)), \
                f"Result contains non-finite values for p={neg_p}"
            
            # Check oracle match
            assert torch.allclose(result_neg, expected_neg, rtol=rtol, atol=atol), \
                f"Result doesn't match oracle for p={neg_p}"
        
        # Test with very small eps to see if negative p handles division by zero
        small_eps = 1e-12
        pairwise_dist_small_eps = nn.PairwiseDistance(p=p, eps=small_eps, keepdim=keepdim)
        result_small_eps = pairwise_dist_small_eps(x1, x2)
        
        # Should still be finite
        assert torch.all(torch.isfinite(result_small_eps)), \
            f"Result should be finite even with small eps={small_eps}"
        
        # Test property: negative p should produce positive values (due to eps in denominator)
        # For p = -1, distance = 1/(|x-y| + eps), which should be positive
        assert torch.all(result > 0), \
            f"Negative p should produce positive values. Got min={result.min().item()}"
        
        # Test with identical vectors
        x1_identical = x1.clone()
        x2_identical = x1.clone()  # Same as x1
        
        result_identical = pairwise_dist(x1_identical, x2_identical)
        # When vectors are identical, |x-y| = 0, so distance = 1/eps for p = -1
        expected_identical = 1.0 / eps
        assert torch.allclose(result_identical, 
                             torch.full_like(result_identical, expected_identical),
                             rtol=1e-5, atol=1e-6), \
            f"Identical vectors with p={p} should give 1/eps={expected_identical}, got {result_identical.mean().item()}"
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:FOOTER START ====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====