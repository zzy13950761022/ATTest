import torch.nn as nn

# Test UpsamplingNearest2d repr
module1 = nn.UpsamplingNearest2d(scale_factor=2.0)
print(f"UpsamplingNearest2d repr: {repr(module1)}")

# Test UpsamplingBilinear2d repr  
module2 = nn.UpsamplingBilinear2d(scale_factor=1.5)
print(f"UpsamplingBilinear2d repr: {repr(module2)}")

# Test with size parameter
module3 = nn.UpsamplingNearest2d(size=(8, 8))
print(f"UpsamplingNearest2d with size repr: {repr(module3)}")

module4 = nn.UpsamplingBilinear2d(size=(6, 6))
print(f"UpsamplingBilinear2d with size repr: {repr(module4)}")