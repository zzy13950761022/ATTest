import torch
import torch.nn as nn
import torch.nn.functional as F

# Test 1: dim=3 for 2D tensor (3, 4)
print("Test 1: dim=3 for 2D tensor (3, 4)")
x1 = torch.randn(3, 4)
x2 = torch.randn(3, 4)

try:
    # Create CosineSimilarity with dim=3
    cosine_sim = nn.CosineSimilarity(dim=3, eps=1e-8)
    result = cosine_sim(x1, x2)
    print(f"  Result shape: {result.shape}")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception type: {type(e).__name__}")
    print(f"  Exception message: {str(e)}")

print("\nTest 2: dim=2 for 2D tensor (3, 4)")
try:
    # Create CosineSimilarity with dim=2
    cosine_sim = nn.CosineSimilarity(dim=2, eps=1e-8)
    result = cosine_sim(x1, x2)
    print(f"  Result shape: {result.shape}")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception type: {type(e).__name__}")
    print(f"  Exception message: {str(e)}")

print("\nTest 3: dim=-1 for 2D tensor (3, 4)")
try:
    # Create CosineSimilarity with dim=-1
    cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)
    result = cosine_sim(x1, x2)
    print(f"  Result shape: {result.shape}")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception type: {type(e).__name__}")
    print(f"  Exception message: {str(e)}")

print("\nTest 4: Direct F.cosine_similarity with dim=3")
try:
    result = F.cosine_similarity(x1, x2, dim=3, eps=1e-8)
    print(f"  Result shape: {result.shape}")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception type: {type(e).__name__}")
    print(f"  Exception message: {str(e)}")