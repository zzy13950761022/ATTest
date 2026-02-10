import torch
from torch.nn.utils import convert_parameters

# Test 1: Vector too short
print("Test 1: Vector too short")
parameters = [torch.randn(2, 2), torch.randn(3)]
total_elements = sum(p.numel() for p in parameters)
print(f"Total elements needed: {total_elements}")

vec_too_short = torch.randn(total_elements - 1)
print(f"Vector length: {vec_too_short.shape[0]}")

target_parameters = [torch.empty_like(p) for p in parameters]

try:
    convert_parameters.vector_to_parameters(vec_too_short, target_parameters)
    print("No exception raised for vector too short!")
except RuntimeError as e:
    print(f"RuntimeError raised: {e}")

# Test 2: Vector too long
print("\nTest 2: Vector too long")
vec_too_long = torch.randn(total_elements + 1)
print(f"Vector length: {vec_too_long.shape[0]}")

target_parameters2 = [torch.empty_like(p) for p in parameters]

try:
    convert_parameters.vector_to_parameters(vec_too_long, target_parameters2)
    print("No exception raised for vector too long!")
except RuntimeError as e:
    print(f"RuntimeError raised: {e}")

# Test 3: Exact length
print("\nTest 3: Exact length")
vec_exact = torch.randn(total_elements)
print(f"Vector length: {vec_exact.shape[0]}")

target_parameters3 = [torch.empty_like(p) for p in parameters]

try:
    convert_parameters.vector_to_parameters(vec_exact, target_parameters3)
    print("Success with exact length vector")
except RuntimeError as e:
    print(f"RuntimeError raised: {e}")