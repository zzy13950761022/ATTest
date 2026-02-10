import torch
import torch.nn as nn

# Test if BatchNorm1d validates num_features
try:
    bn = nn.BatchNorm1d(num_features=0)
    print("BatchNorm1d with num_features=0 did NOT raise error")
except Exception as e:
    print(f"BatchNorm1d with num_features=0 raised: {type(e).__name__}: {e}")

try:
    bn = nn.BatchNorm1d(num_features=-1)
    print("BatchNorm1d with num_features=-1 did NOT raise error")
except Exception as e:
    print(f"BatchNorm1d with num_features=-1 raised: {type(e).__name__}: {e}")

# Test if BatchNorm1d validates eps
try:
    bn = nn.BatchNorm1d(num_features=10, eps=0.0)
    print("BatchNorm1d with eps=0.0 did NOT raise error")
except Exception as e:
    print(f"BatchNorm1d with eps=0.0 raised: {type(e).__name__}: {e}")

try:
    bn = nn.BatchNorm1d(num_features=10, eps=-1e-5)
    print("BatchNorm1d with eps=-1e-5 did NOT raise error")
except Exception as e:
    print(f"BatchNorm1d with eps=-1e-5 raised: {type(e).__name__}: {e}")

# Test if BatchNorm1d validates momentum
try:
    bn = nn.BatchNorm1d(num_features=10, momentum=1.1)
    print("BatchNorm1d with momentum=1.1 did NOT raise error")
except Exception as e:
    print(f"BatchNorm1d with momentum=1.1 raised: {type(e).__name__}: {e}")

try:
    bn = nn.BatchNorm1d(num_features=10, momentum=-0.1)
    print("BatchNorm1d with momentum=-0.1 did NOT raise error")
except Exception as e:
    print(f"BatchNorm1d with momentum=-0.1 raised: {type(e).__name__}: {e}")