import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

# Test clip_grad_norm_ with invalid max_norm
tensor = torch.randn(2, 3, requires_grad=True)
tensor.grad = torch.randn_like(tensor)

print("Testing clip_grad_norm_ with max_norm=0...")
try:
    result = clip_grad_norm_(tensor, max_norm=0)
    print(f"Result: {result}")
    print("No error raised!")
except RuntimeError as e:
    print(f"RuntimeError raised: {e}")
except Exception as e:
    print(f"Other error raised: {type(e).__name__}: {e}")

print("\nTesting clip_grad_norm_ with max_norm=-1.0...")
try:
    result = clip_grad_norm_(tensor, max_norm=-1.0)
    print(f"Result: {result}")
    print("No error raised!")
except RuntimeError as e:
    print(f"RuntimeError raised: {e}")
except Exception as e:
    print(f"Other error raised: {type(e).__name__}: {e}")

print("\nTesting clip_grad_value_ with clip_value=0...")
try:
    result = clip_grad_value_(tensor, clip_value=0)
    print(f"Result: {result}")
    print("No error raised!")
except RuntimeError as e:
    print(f"RuntimeError raised: {e}")
except Exception as e:
    print(f"Other error raised: {type(e).__name__}: {e}")

print("\nTesting clip_grad_value_ with clip_value=-1.0...")
try:
    result = clip_grad_value_(tensor, clip_value=-1.0)
    print(f"Result: {result}")
    print("No error raised!")
except RuntimeError as e:
    print(f"RuntimeError raised: {e}")
except Exception as e:
    print(f"Other error raised: {type(e).__name__}: {e}")