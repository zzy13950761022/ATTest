import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# 测试 eps=0 的情况
linear = nn.Linear(10, 20)
try:
    sn_linear = spectral_norm(linear, name='weight', n_power_iterations=1, eps=0, dim=None)
    print("eps=0: SUCCESS - No exception raised")
    print(f"eps value in module: {sn_linear._forward_pre_hooks[list(sn_linear._forward_pre_hooks.keys())[0]].eps}")
except Exception as e:
    print(f"eps=0: FAILED - {type(e).__name__}: {e}")

# 测试 eps=-1e-12 的情况
linear2 = nn.Linear(10, 20)
try:
    sn_linear2 = spectral_norm(linear2, name='weight', n_power_iterations=1, eps=-1e-12, dim=None)
    print("eps=-1e-12: SUCCESS - No exception raised")
    print(f"eps value in module: {sn_linear2._forward_pre_hooks[list(sn_linear2._forward_pre_hooks.keys())[0]].eps}")
except Exception as e:
    print(f"eps=-1e-12: FAILED - {type(e).__name__}: {e}")

# 测试 dim=2 的情况（线性层权重是2维）
linear3 = nn.Linear(10, 20)
try:
    sn_linear3 = spectral_norm(linear3, name='weight', n_power_iterations=1, eps=1e-12, dim=2)
    print("dim=2: SUCCESS - No exception raised")
except Exception as e:
    print(f"dim=2: FAILED - {type(e).__name__}: {e}")

# 测试 dim=-3 的情况
linear4 = nn.Linear(10, 20)
try:
    sn_linear4 = spectral_norm(linear4, name='weight', n_power_iterations=1, eps=1e-12, dim=-3)
    print("dim=-3: SUCCESS - No exception raised")
except Exception as e:
    print(f"dim=-3: FAILED - {type(e).__name__}: {e}")