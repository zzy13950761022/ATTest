import torch
from torch.nn.utils import convert_parameters

# Create parameters
p1 = torch.randn(2, 2)  # 4 elements
p2 = torch.randn(3)     # 3 elements
parameters = [p1, p2]
total_elements = 4 + 3  # 7 elements

print(f"Total elements needed: {total_elements}")

# Test with vector that's too short (6 elements instead of 7)
vec_short = torch.randn(6)
print(f"\nTest 1: Vector too short (length {vec_short.shape[0]})")
target1 = [torch.empty_like(p1), torch.empty_like(p2)]

try:
    convert_parameters.vector_to_parameters(vec_short, target1)
    print("No exception raised!")
    # Check what happened
    print(f"First parameter shape: {target1[0].shape}, numel: {target1[0].numel()}")
    print(f"Second parameter shape: {target1[1].shape}, numel: {target1[1].numel()}")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
except Exception as e:
    print(f"Other exception ({type(e).__name__}): {e}")

# Test with vector that's too long (8 elements instead of 7)
vec_long = torch.randn(8)
print(f"\nTest 2: Vector too long (length {vec_long.shape[0]})")
target2 = [torch.empty_like(p1), torch.empty_like(p2)]

try:
    convert_parameters.vector_to_parameters(vec_long, target2)
    print("No exception raised!")
    # Check what happened
    print(f"First parameter shape: {target2[0].shape}, numel: {target2[0].numel()}")
    print(f"Second parameter shape: {target2[1].shape}, numel: {target2[1].numel()}")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
except Exception as e:
    print(f"Other exception ({type(e).__name__}): {e}")