import torch
import torch.nn.init as init

# 测试 sparse_ 对非法稀疏度的行为
print("测试 sparse_ 函数对非法稀疏度的行为...")

# 测试稀疏度 > 1
tensor = torch.empty(4, 4)
print(f"测试稀疏度 1.5 (大于1)...")
try:
    result = init.sparse_(tensor, sparsity=1.5)
    print(f"  结果: 没有抛出异常，稀疏度 1.5 被接受")
    print(f"  零元素数量: {(result == 0).sum().item()}")
    print(f"  总元素数量: {result.numel()}")
except Exception as e:
    print(f"  异常类型: {type(e).__name__}")
    print(f"  异常消息: {e}")

print("\n测试稀疏度 2.0 (等于2)...")
tensor2 = torch.empty(4, 4)
try:
    result2 = init.sparse_(tensor2, sparsity=2.0)
    print(f"  结果: 没有抛出异常，稀疏度 2.0 被接受")
    print(f"  零元素数量: {(result2 == 0).sum().item()}")
    print(f"  总元素数量: {result2.numel()}")
except Exception as e:
    print(f"  异常类型: {type(e).__name__}")
    print(f"  异常消息: {e}")

print("\n测试稀疏度 -0.5 (小于0)...")
tensor3 = torch.empty(4, 4)
try:
    result3 = init.sparse_(tensor3, sparsity=-0.5)
    print(f"  结果: 没有抛出异常，稀疏度 -0.5 被接受")
    print(f"  零元素数量: {(result3 == 0).sum().item()}")
    print(f"  总元素数量: {result3.numel()}")
except Exception as e:
    print(f"  异常类型: {type(e).__name__}")
    print(f"  异常消息: {e}")

print("\n测试稀疏度 0.0 (等于0)...")
tensor4 = torch.empty(4, 4)
try:
    result4 = init.sparse_(tensor4, sparsity=0.0)
    print(f"  结果: 没有抛出异常，稀疏度 0.0 被接受")
    print(f"  零元素数量: {(result4 == 0).sum().item()}")
    print(f"  总元素数量: {result4.numel()}")
except Exception as e:
    print(f"  异常类型: {type(e).__name__}")
    print(f"  异常消息: {e}")

print("\n测试稀疏度 1.0 (等于1)...")
tensor5 = torch.empty(4, 4)
try:
    result5 = init.sparse_(tensor5, sparsity=1.0)
    print(f"  结果: 没有抛出异常，稀疏度 1.0 被接受")
    print(f"  零元素数量: {(result5 == 0).sum().item()}")
    print(f"  总元素数量: {result5.numel()}")
except Exception as e:
    print(f"  异常类型: {type(e).__name__}")
    print(f"  异常消息: {e}")