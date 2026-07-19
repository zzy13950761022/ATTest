import torch
import torch.nn as nn

print("Testing BatchNorm1d parameter validation...")

# Test if BatchNorm1d validates num_features
print("\n1. Testing num_features validation:")
try:
    bn = nn.BatchNorm1d(num_features=0)
    print("  BatchNorm1d with num_features=0: SUCCESS (no error)")
except Exception as e:
    print(f"  BatchNorm1d with num_features=0: {type(e).__name__}: {e}")

try:
    bn = nn.BatchNorm1d(num_features=-1)
    print("  BatchNorm1d with num_features=-1: SUCCESS (no error)")
except Exception as e:
    print(f"  BatchNorm1d with num_features=-1: {type(e).__name__}: {e}")

# Test if BatchNorm1d validates eps
print("\n2. Testing eps validation:")
try:
    bn = nn.BatchNorm1d(num_features=10, eps=0.0)
    print("  BatchNorm1d with eps=0.0: SUCCESS (no error)")
except Exception as e:
    print(f"  BatchNorm1d with eps=0.0: {type(e).__name__}: {e}")

try:
    bn = nn.BatchNorm1d(num_features=10, eps=-1e-5)
    print("  BatchNorm1d with eps=-1e-5: SUCCESS (no error)")
except Exception as e:
    print(f"  BatchNorm1d with eps=-1e-5: {type(e).__name__}: {e}")

# Test if BatchNorm1d validates momentum
print("\n3. Testing momentum validation:")
try:
    bn = nn.BatchNorm1d(num_features=10, momentum=1.1)
    print("  BatchNorm1d with momentum=1.1: SUCCESS (no error)")
except Exception as e:
    print(f"  BatchNorm1d with momentum=1.1: {type(e).__name__}: {e}")

try:
    bn = nn.BatchNorm1d(num_features=10, momentum=-0.1)
    print("  BatchNorm1d with momentum=-0.1: SUCCESS (no error)")
except Exception as e:
    print(f"  BatchNorm1d with momentum=-0.1: {type(e).__name__}: {e}")

# Test input dimension validation
print("\n4. Testing input dimension validation:")
bn1d = nn.BatchNorm1d(num_features=10)
bn2d = nn.BatchNorm2d(num_features=10)
bn3d = nn.BatchNorm3d(num_features=10)

input_2d = torch.randn(4, 10)
input_3d = torch.randn(4, 10, 32)
input_4d = torch.randn(4, 10, 16, 16)

print("  Testing BatchNorm1d with 2D input:")
try:
    output = bn1d(input_2d)
    print("    SUCCESS")
except Exception as e:
    print(f"    {type(e).__name__}: {e}")

print("  Testing BatchNorm1d with 3D input:")
try:
    output = bn1d(input_3d)
    print("    SUCCESS")
except Exception as e:
    print(f"    {type(e).__name__}: {e}")

print("  Testing BatchNorm2d with 2D input (should fail):")
try:
    output = bn2d(input_2d)
    print("    SUCCESS (unexpected!)")
except Exception as e:
    print(f"    {type(e).__name__}: {e}")

print("  Testing BatchNorm3d with 2D input (should fail):")
try:
    output = bn3d(input_2d)
    print("    SUCCESS (unexpected!)")
except Exception as e:
    print(f"    {type(e).__name__}: {e}")