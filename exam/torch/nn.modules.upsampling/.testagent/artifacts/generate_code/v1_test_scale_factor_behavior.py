import torch
import torch.nn as nn

# Test negative scale factor
print("Testing negative scale factor...")
try:
    upsample = nn.Upsample(scale_factor=-1.0, mode='nearest')
    x = torch.randn(1, 3, 4, 4)
    output = upsample(x)
    print(f"Negative scale_factor worked! Output shape: {output.shape}")
except Exception as e:
    print(f"Negative scale_factor failed with: {type(e).__name__}: {e}")

# Test zero scale factor
print("\nTesting zero scale factor...")
try:
    upsample = nn.Upsample(scale_factor=0.0, mode='nearest')
    x = torch.randn(1, 3, 4, 4)
    output = upsample(x)
    print(f"Zero scale_factor worked! Output shape: {output.shape}")
except Exception as e:
    print(f"Zero scale_factor failed with: {type(e).__name__}: {e}")

# Test very small positive scale factor
print("\nTesting scale_factor=0.5...")
try:
    upsample = nn.Upsample(scale_factor=0.5, mode='nearest')
    x = torch.randn(1, 3, 4, 4)
    output = upsample(x)
    print(f"scale_factor=0.5 worked! Output shape: {output.shape}")
except Exception as e:
    print(f"scale_factor=0.5 failed with: {type(e).__name__}: {e}")

# Test what happens with floor calculation
print("\nTesting floor calculation for scale_factor=0.5 with 4x4 input...")
x = torch.randn(1, 3, 4, 4)
upsample = nn.Upsample(scale_factor=0.5, mode='nearest')
output = upsample(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
print(f"Expected output size: floor(4 * 0.5) = {int(4 * 0.5)}")