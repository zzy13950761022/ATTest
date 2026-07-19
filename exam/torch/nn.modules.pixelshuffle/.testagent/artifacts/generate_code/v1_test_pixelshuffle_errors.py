import torch
import torch.nn as nn

# Test PixelShuffle with invalid upscale_factor
print("Testing PixelShuffle error messages:")
try:
    ps = nn.PixelShuffle(0)
    input_tensor = torch.randn(1, 4, 8, 8)
    output = ps(input_tensor)
except RuntimeError as e:
    print(f"PixelShuffle(0) error: {e}")

try:
    ps = nn.PixelShuffle(-1)
    input_tensor = torch.randn(1, 4, 8, 8)
    output = ps(input_tensor)
except RuntimeError as e:
    print(f"PixelShuffle(-1) error: {e}")

# Test PixelShuffle with non-divisible channels
try:
    ps = nn.PixelShuffle(2)
    input_tensor = torch.randn(1, 15, 4, 4)  # 15 not divisible by 4
    output = ps(input_tensor)
except RuntimeError as e:
    print(f"PixelShuffle non-divisible channels error: {e}")

# Test PixelUnshuffle error messages for comparison
print("\nTesting PixelUnshuffle error messages:")
try:
    pu = nn.PixelUnshuffle(0)
    input_tensor = torch.randn(1, 4, 8, 8)
    output = pu(input_tensor)
except RuntimeError as e:
    print(f"PixelUnshuffle(0) error: {e}")

try:
    pu = nn.PixelUnshuffle(2)
    input_tensor = torch.randn(1, 4, 5, 5)  # 5 not divisible by 2
    output = pu(input_tensor)
except RuntimeError as e:
    print(f"PixelUnshuffle non-divisible dimensions error: {e}")