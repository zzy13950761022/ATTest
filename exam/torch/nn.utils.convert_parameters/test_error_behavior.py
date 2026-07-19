import torch
from torch.nn.utils import convert_parameters

# Test 1: Vector shorter than needed
print("Test 1: Vector shorter than needed")
parameters = [torch.empty(2, 2), torch.empty(3)]
vec = torch.randn(5)  # Need 7 elements, only have 5

try:
    convert_parameters.vector_to_parameters(vec, parameters)
except RuntimeError as e:
    print(f"Error message: {e}")

# Test 2: Vector exactly the right length
print("\nTest 2: Vector exactly the right length")
parameters = [torch.empty(2, 2), torch.empty(3)]
vec = torch.randn(7)  # Exactly 7 elements

try:
    convert_parameters.vector_to_parameters(vec, parameters)
    print("Success: No error with correct length")
except RuntimeError as e:
    print(f"Error message: {e}")

# Test 3: Vector longer than needed
print("\nTest 3: Vector longer than needed")
parameters = [torch.empty(2, 2), torch.empty(3)]
vec = torch.randn(10)  # 10 elements, need 7

try:
    convert_parameters.vector_to_parameters(vec, parameters)
except RuntimeError as e:
    print(f"Error message: {e}")

# Test 4: Let's trace what happens
print("\nTest 4: Manual tracing")
parameters = [torch.empty(2, 2), torch.empty(3)]
vec = torch.randn(5)  # Only 5 elements

print(f"Total elements needed: {4 + 3} = 7")
print(f"Vector length: {len(vec)} = 5")

# Simulate what happens in vector_to_parameters
pointer = 0
for i, param in enumerate(parameters):
    num_param = param.numel()
    print(f"\nParameter {i}: shape {param.shape}, numel {num_param}")
    print(f"Pointer: {pointer}, Slice: vec[{pointer}:{pointer + num_param}]")
    
    if pointer + num_param > len(vec):
        print(f"ERROR: Not enough elements in vector!")
        print(f"  Need {num_param} elements, but only {len(vec) - pointer} available")
        # Try to slice anyway to see what error we get
        try:
            slice = vec[pointer:pointer + num_param]
            print(f"  Slice shape: {slice.shape}")
            reshaped = slice.view_as(param)
        except RuntimeError as e:
            print(f"  Error when viewing: {e}")
    
    pointer += num_param