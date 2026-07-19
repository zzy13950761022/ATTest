import torch
from torch.nn.utils import convert_parameters

# Test vector length mismatch
try:
    # Create a vector with 5 elements
    vec = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    
    # Create parameters that need 6 elements total
    param1 = torch.empty(2, 2, dtype=torch.float32)  # 4 elements
    param2 = torch.empty(3, dtype=torch.float32)     # 3 elements, total 7 > 5
    
    convert_parameters.vector_to_parameters(vec, [param1, param2])
    print("No error raised - unexpected!")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    
# Test another scenario
try:
    # Create a vector with 7 elements
    vec = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float32)
    
    # Create parameters that need 6 elements total
    param1 = torch.empty(2, 2, dtype=torch.float32)  # 4 elements
    param2 = torch.empty(3, dtype=torch.float32)     # 3 elements, total 7
    
    convert_parameters.vector_to_parameters(vec, [param1, param2])
    print("\nSecond test: No error raised - expected!")
except Exception as e:
    print(f"\nSecond test - Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")