import torch
import torch.nn as nn

# Test PixelShuffle with invalid upscale_factor
print("Testing PixelShuffle with invalid upscale_factor:")
try:
    ps = nn.PixelShuffle(0)
    print("PixelShuffle(0) created successfully")
except Exception as e:
    print(f"PixelShuffle(0) raised: {type(e).__name__}: {e}")

try:
    ps = nn.PixelShuffle(-1)
    print("PixelShuffle(-1) created successfully")
except Exception as e:
    print(f"PixelShuffle(-1) raised: {type(e).__name__}: {e}")

# Test PixelUnshuffle with invalid downscale_factor
print("\nTesting PixelUnshuffle with invalid downscale_factor:")
try:
    pu = nn.PixelUnshuffle(0)
    print("PixelUnshuffle(0) created successfully")
    # Test forward pass
    input_tensor = torch.randn(1, 4, 8, 8)
    output = pu(input_tensor)
    print(f"Forward pass succeeded: {output.shape}")
except Exception as e:
    print(f"PixelUnshuffle(0) raised: {type(e).__name__}: {e}")

try:
    pu = nn.PixelUnshuffle(-1)
    print("PixelUnshuffle(-1) created successfully")
    # Test forward pass
    input_tensor = torch.randn(1, 4, 8, 8)
    output = pu(input_tensor)
    print(f"Forward pass succeeded: {output.shape}")
except Exception as e:
    print(f"PixelUnshuffle(-1) raised: {type(e).__name__}: {e}")

# Test actual error messages
print("\nTesting actual error messages:")
pu2 = nn.PixelUnshuffle(2)
input_bad = torch.randn(1, 4, 5, 5)  # 5 not divisible by 2
try:
    output = pu2(input_bad)
except RuntimeError as e:
    print(f"Actual error message for non-divisible dimensions: {e}")