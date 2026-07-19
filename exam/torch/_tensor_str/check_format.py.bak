import torch

# Test basic float tensor
tensor1 = torch.tensor([[0., 1., 2.], [3., 4., 5.]], dtype=torch.float32)
print("Basic float tensor:")
print(str(tensor1))
print()

# Test sparse tensor
dense = torch.randn(3, 3)
mask = torch.rand(3, 3) > 0.5
sparse_tensor = dense.sparse_mask(mask.to_sparse_coo())
print("Sparse tensor:")
print(str(sparse_tensor))
print()

# Test complex tensor
complex_tensor = torch.complex(
    torch.tensor([[0., 1.], [2., 3.]]),
    torch.tensor([[0., 0.5], [1., 1.5]])
)
print("Complex tensor:")
print(str(complex_tensor))
print()

# Test with dtype at the end
print("Checking if dtype= appears in output:")
print(f"Basic float: {'dtype=' in str(tensor1)}")
print(f"Sparse: {'dtype=' in str(sparse_tensor)}")
print(f"Complex: {'dtype=' in str(complex_tensor)}")