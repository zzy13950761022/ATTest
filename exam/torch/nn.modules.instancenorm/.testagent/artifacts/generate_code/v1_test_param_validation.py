import torch
import torch.nn as nn

print("Testing InstanceNorm1d parameter validation...")

# Test 1: eps parameter validation
print("\n1. Testing eps parameter:")
try:
    norm = nn.InstanceNorm1d(num_features=3, eps=0.0)
    print(f"  eps=0.0: Created successfully, eps={norm.eps}")
except ValueError as e:
    print(f"  eps=0.0: ValueError: {e}")

try:
    norm = nn.InstanceNorm1d(num_features=3, eps=-1.0)
    print(f"  eps=-1.0: Created successfully, eps={norm.eps}")
except ValueError as e:
    print(f"  eps=-1.0: ValueError: {e}")

try:
    norm = nn.InstanceNorm1d(num_features=3, eps=1e-5)
    print(f"  eps=1e-5: Created successfully, eps={norm.eps}")
except ValueError as e:
    print(f"  eps=1e-5: ValueError: {e}")

# Test 2: momentum parameter validation
print("\n2. Testing momentum parameter:")
try:
    norm = nn.InstanceNorm1d(num_features=3, momentum=-0.1)
    print(f"  momentum=-0.1: Created successfully, momentum={norm.momentum}")
except ValueError as e:
    print(f"  momentum=-0.1: ValueError: {e}")

try:
    norm = nn.InstanceNorm1d(num_features=3, momentum=1.1)
    print(f"  momentum=1.1: Created successfully, momentum={norm.momentum}")
except ValueError as e:
    print(f"  momentum=1.1: ValueError: {e}")

try:
    norm = nn.InstanceNorm1d(num_features=3, momentum=0.0)
    print(f"  momentum=0.0: Created successfully, momentum={norm.momentum}")
except ValueError as e:
    print(f"  momentum=0.0: ValueError: {e}")

try:
    norm = nn.InstanceNorm1d(num_features=3, momentum=1.0)
    print(f"  momentum=1.0: Created successfully, momentum={norm.momentum}")
except ValueError as e:
    print(f"  momentum=1.0: ValueError: {e}")

# Test 3: Channel mismatch
print("\n3. Testing channel dimension mismatch:")
norm = nn.InstanceNorm2d(num_features=3)
input_wrong = torch.randn(2, 4, 5, 5)  # 4 channels, expected 3
try:
    output = norm(input_wrong)
    print(f"  Channel mismatch: Forward pass succeeded (unexpected)")
except RuntimeError as e:
    print(f"  Channel mismatch: RuntimeError: {e}")
except Exception as e:
    print(f"  Channel mismatch: {type(e).__name__}: {e}")

# Test 4: LazyInstanceNorm parameter validation
print("\n4. Testing LazyInstanceNorm2d parameter validation:")
try:
    lazy_norm = nn.LazyInstanceNorm2d(momentum=-0.1)
    print(f"  momentum=-0.1: Created successfully, momentum={lazy_norm.momentum}")
except ValueError as e:
    print(f"  momentum=-0.1: ValueError: {e}")

try:
    lazy_norm = nn.LazyInstanceNorm2d(momentum=1.1)
    print(f"  momentum=1.1: Created successfully, momentum={lazy_norm.momentum}")
except ValueError as e:
    print(f"  momentum=1.1: ValueError: {e}")

print("\nDone.")