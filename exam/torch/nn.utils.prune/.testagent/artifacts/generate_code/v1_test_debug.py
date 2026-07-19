import torch
import torch.nn as nn
from torch.nn.utils import prune

# 创建一个简单的测试来理解剪枝行为
module = nn.Linear(3, 3)
param_name = "weight"

print("初始权重形状:", module.weight.shape)
print("初始权重值:", module.weight)

# 克隆原始权重
original_weight = module.weight.clone()
print("\n克隆的原始权重:", original_weight)

# 修改权重
with torch.no_grad():
    flat_weights = module.weight.view(-1)
    for i in range(len(flat_weights)):
        flat_weights[i] = i * 0.1

print("\n修改后的模块权重:", module.weight)
print("克隆的原始权重（应该未改变）:", original_weight)

# 应用剪枝
pruned_module = prune.l1_unstructured(module, param_name, amount=2)

print("\n剪枝后的模块:")
print("权重:", getattr(pruned_module, param_name))
print("原始参数:", getattr(pruned_module, f"{param_name}_orig"))
print("掩码:", getattr(pruned_module, f"{param_name}_mask"))

# 检查原始参数是否等于修改后的权重
orig_param = getattr(pruned_module, f"{param_name}_orig")
print("\n原始参数是否等于修改后的权重?", torch.allclose(orig_param, module.weight))
print("原始参数是否等于克隆的原始权重?", torch.allclose(orig_param, original_weight))