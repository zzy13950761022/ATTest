import torch
from torch.nn.utils import clip_grad_norm_

# Test actual behavior with max_norm <= 0
tensor = torch.randn(2, 3, requires_grad=True)
tensor.grad = torch.randn_like(tensor)

print("Testing with max_norm = 0:")
try:
    result = clip_grad_norm_(tensor, max_norm=0)
    print(f"  No error raised. Result: {result}")
except Exception as e:
    print(f"  Error raised: {type(e).__name__}: {e}")

print("\nTesting with max_norm = -1.0:")
try:
    result = clip_grad_norm_(tensor, max_norm=-1.0)
    print(f"  No error raised. Result: {result}")
except Exception as e:
    print(f"  Error raised: {type(e).__name__}: {e}")

print("\nTesting with max_norm = 0.5 (positive):")
try:
    result = clip_grad_norm_(tensor, max_norm=0.5)
    print(f"  Success. Result: {result}")
except Exception as e:
    print(f"  Error raised: {type(e).__name__}: {e}")