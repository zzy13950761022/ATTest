import math
import pytest
import torch
import torch.nn.init as init
from typing import Tuple, List, Any


# ==== BLOCK:HEADER START ====
import math
import pytest
import torch
import torch.nn.init as init
from typing import Tuple, List, Any


def set_random_seed(seed: int = 42) -> None:
    """设置随机种子以确保测试可重复性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_tensor_properties(tensor: torch.Tensor, 
                            expected_shape: Tuple[int, ...],
                            expected_dtype: torch.dtype,
                            test_name: str = "") -> None:
    """验证张量的基本属性"""
    assert tensor.shape == expected_shape, \
        f"{test_name}: 形状不匹配，期望 {expected_shape}，实际 {tensor.shape}"
    assert tensor.dtype == expected_dtype, \
        f"{test_name}: 数据类型不匹配，期望 {expected_dtype}，实际 {tensor.dtype}"
    assert torch.isfinite(tensor).all(), \
        f"{test_name}: 张量包含非有限值"


def assert_not_all_zero(tensor: torch.Tensor, test_name: str = "") -> None:
    """验证张量不全为零"""
    assert not torch.all(tensor == 0), \
        f"{test_name}: 张量全为零"


def assert_eye_pattern(tensor: torch.Tensor, test_name: str = "") -> None:
    """验证张量具有单位矩阵模式"""
    n, m = tensor.shape[-2], tensor.shape[-1]
    min_dim = min(n, m)
    
    # 检查对角线元素接近1
    for i in range(min_dim):
        diag_value = tensor[..., i, i].mean().item()
        assert abs(diag_value - 1.0) < 1e-6, \
            f"{test_name}: 对角线元素 {i},{i} 应该是1，实际 {diag_value}"
    
    # 检查非对角线元素接近0
    zero_mask = torch.ones_like(tensor, dtype=torch.bool)
    for i in range(min_dim):
        zero_mask[..., i, i] = False
    
    off_diag_values = tensor[zero_mask]
    if off_diag_values.numel() > 0:
        max_off_diag = torch.abs(off_diag_values).max().item()
        assert max_off_diag < 1e-6, \
            f"{test_name}: 非对角线元素最大绝对值 {max_off_diag} 应该接近0"


# 设置全局随机种子
set_random_seed(42)
# ==== BLOCK:HEADER END ====


# ==== BLOCK:CASE_09 START ====
# eye_和dirac_基础测试（占位）
# ==== BLOCK:CASE_09 END ====


# ==== BLOCK:CASE_10 START ====
# sparse_和orthogonal_测试（占位）
# ==== BLOCK:CASE_10 END ====


# ==== BLOCK:CASE_11 START ====
# 特殊函数维度边界（占位）
# ==== BLOCK:CASE_11 END ====


# ==== BLOCK:FOOTER START ====
def test_invalid_inputs() -> None:
    """测试非法输入场景"""
    
    # 测试 eye_ 的非法维度
    tensor_1d = torch.empty(3)
    with pytest.raises(ValueError, match="Only tensors with 2 dimensions are supported"):
        init.eye_(tensor_1d)
    
    # 测试 dirac_ 的非法维度
    tensor_2d = torch.empty(3, 3)
    with pytest.raises(ValueError, match="Only tensors with 3, 4, or 5 dimensions are supported"):
        init.dirac_(tensor_2d)
    
    # 测试 sparse_ 的非法稀疏度
    tensor = torch.empty(4, 4)
    with pytest.raises(RuntimeError):
        init.sparse_(tensor, sparsity=1.5)  # 稀疏度大于1


def test_no_grad_context() -> None:
    """验证初始化函数在无梯度上下文中工作"""
    tensor = torch.empty(3, 3, requires_grad=True)
    
    # 保存原始梯度状态
    original_requires_grad = tensor.requires_grad
    
    # 调用初始化函数
    result = init.eye_(tensor)
    
    # 验证梯度状态保持不变
    assert result.requires_grad == original_requires_grad, \
        "初始化不应该改变张量的梯度状态"
    
    # 验证张量被修改了
    assert not torch.allclose(result, torch.zeros_like(result)), \
        "张量应该被修改"


def test_random_seed_consistency() -> None:
    """验证随机种子的一致性（对于随机初始化函数）"""
    shape = (4, 4)
    
    # 测试 sparse_ 的随机种子一致性
    set_random_seed(42)
    tensor1 = torch.empty(shape)
    result1 = init.sparse_(tensor1, sparsity=0.5)
    
    set_random_seed(42)
    tensor2 = torch.empty(shape)
    result2 = init.sparse_(tensor2, sparsity=0.5)
    
    # 验证稀疏模式相同
    zero_mask1 = result1 == 0
    zero_mask2 = result2 == 0
    assert torch.all(zero_mask1 == zero_mask2), \
        "相同随机种子应该产生相同的稀疏模式"


if __name__ == "__main__":
    # 简单的手动测试
    print("运行G3组手动测试...")
    
    # 测试 eye_
    tensor = torch.empty(3, 3)
    result = init.eye_(tensor)
    print(f"eye_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 dirac_ (需要3-5维)
    tensor = torch.empty(3, 3, 3, 3)  # 4维
    result = init.dirac_(tensor)
    print(f"dirac_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 sparse_
    tensor = torch.empty(4, 4)
    result = init.sparse_(tensor, sparsity=0.5)
    print(f"sparse_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 orthogonal_
    tensor = torch.empty(4, 4)
    result = init.orthogonal_(tensor)
    print(f"orthogonal_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    print("G3组所有手动测试完成！")
# ==== BLOCK:FOOTER END ====