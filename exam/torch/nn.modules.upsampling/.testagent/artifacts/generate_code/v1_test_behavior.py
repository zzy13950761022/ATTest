import torch
import torch.nn as nn

# 测试PyTorch实际行为
x = torch.randn(1, 3, 4, 4)

print("测试1: 同时指定size和scale_factor")
try:
    upsample = nn.Upsample(size=[8, 8], scale_factor=2.0, mode='nearest')
    print("  初始化成功")
    output = upsample(x)
    print(f"  forward成功, 输出形状: {output.shape}")
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")

print("\n测试2: 只指定size")
try:
    upsample = nn.Upsample(size=[8, 8], mode='nearest')
    output = upsample(x)
    print(f"  成功, 输出形状: {output.shape}")
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")

print("\n测试3: 只指定scale_factor")
try:
    upsample = nn.Upsample(scale_factor=2.0, mode='nearest')
    output = upsample(x)
    print(f"  成功, 输出形状: {output.shape}")
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")

print("\n测试4: 都不指定")
try:
    upsample = nn.Upsample(mode='nearest')
    output = upsample(x)
    print(f"  成功, 输出形状: {output.shape}")
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")

print("\n测试5: 测试F.interpolate的行为")
import torch.nn.functional as F

print("  同时指定size和scale_factor:")
try:
    output = F.interpolate(x, size=[8, 8], scale_factor=2.0, mode='nearest')
    print(f"    成功, 输出形状: {output.shape}")
except Exception as e:
    print(f"    错误: {type(e).__name__}: {e}")

print("\n测试6: 测试UpsamplingNearest2d")
try:
    upsample = nn.UpsamplingNearest2d(size=(8, 8), scale_factor=2.0)
    print("  初始化成功")
    output = upsample(x)
    print(f"  forward成功, 输出形状: {output.shape}")
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")