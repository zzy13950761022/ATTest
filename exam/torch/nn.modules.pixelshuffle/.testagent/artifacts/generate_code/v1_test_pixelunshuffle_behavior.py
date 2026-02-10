import torch
import torch.nn as nn

# Test PixelUnshuffle with invalid parameters
try:
    pixel_unshuffle = nn.PixelUnshuffle(0)
    print("PixelUnshuffle(0) did not raise ValueError")
except ValueError as e:
    print(f"PixelUnshuffle(0) raised ValueError: {e}")

try:
    pixel_unshuffle = nn.PixelUnshuffle(-1)
    print("PixelUnshuffle(-1) did not raise ValueError")
except ValueError as e:
    print(f"PixelUnshuffle(-1) raised ValueError: {e}")

try:
    pixel_unshuffle = nn.PixelUnshuffle(1)
    print("PixelUnshuffle(1) created successfully")
except Exception as e:
    print(f"PixelUnshuffle(1) raised {type(e).__name__}: {e}")

# Test with valid parameters
pixel_unshuffle = nn.PixelUnshuffle(2)
print(f"PixelUnshuffle(2) created successfully")

# Test forward with invalid dimensions
input_tensor = torch.randn(1, 4, 5, 5)  # 5 not divisible by 2
try:
    output = pixel_unshuffle(input_tensor)
    print("Forward with invalid dimensions did not raise RuntimeError")
except RuntimeError as e:
    print(f"Forward with invalid dimensions raised RuntimeError: {e}")