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


def assert_in_range(tensor: torch.Tensor, 
                   min_val: float, 
                   max_val: float,
                   test_name: str = "") -> None:
    """验证张量值在指定范围内"""
    assert (tensor >= min_val).all(), \
        f"{test_name}: 有值小于 {min_val}"
    assert (tensor <= max_val).all(), \
        f"{test_name}: 有值大于 {max_val}"


def assert_not_all_zero(tensor: torch.Tensor, test_name: str = "") -> None:
    """验证张量不全为零"""
    assert not torch.all(tensor == 0), \
        f"{test_name}: 张量全为零"


def assert_all_equal(tensor: torch.Tensor, 
                    expected_value: float,
                    test_name: str = "",
                    rtol: float = 1e-6) -> None:
    """验证张量所有元素等于指定值"""
    assert torch.allclose(tensor, 
                         torch.full_like(tensor, expected_value),
                         rtol=rtol), \
        f"{test_name}: 张量元素不全等于 {expected_value}"


# 设置全局随机种子
set_random_seed(42)
# ==== BLOCK:HEADER END ====


# ==== BLOCK:CASE_01 START ====
# uniform_基础测试（占位）
# ==== BLOCK:CASE_01 END ====


# ==== BLOCK:CASE_02 START ====
# normal_基础测试（占位）
# ==== BLOCK:CASE_02 END ====


# ==== BLOCK:CASE_03 START ====
# constant_和ones_zeros_测试（占位）
# ==== BLOCK:CASE_03 END ====


# ==== BLOCK:CASE_04 START ====
# 基础函数边界测试（占位）
# ==== BLOCK:CASE_04 END ====


# ==== BLOCK:FOOTER START ====
def test_invalid_inputs() -> None:
    """测试非法输入场景"""
    
    # 测试 uniform_ 的非法输入
    with pytest.raises(RuntimeError):
        # 非张量输入
        init.uniform_(None, 0, 1)
    
    # 测试 normal_ 的非法标准差
    tensor = torch.empty(2, 3)
    with pytest.raises(RuntimeError):
        init.normal_(tensor, mean=0, std=-1)  # 负标准差
    
    # 测试 constant_ 的非法值
    tensor = torch.empty(2, 3)
    # constant_ 应该接受任何浮点值，这里测试正常情况
    result = init.constant_(tensor, 5.0)
    assert torch.allclose(result, torch.full_like(result, 5.0))


def test_no_grad_context() -> None:
    """验证初始化函数在无梯度上下文中工作"""
    tensor = torch.empty(3, 4, requires_grad=True)
    
    # 保存原始梯度状态
    original_requires_grad = tensor.requires_grad
    
    # 调用初始化函数
    result = init.uniform_(tensor, 0, 1)
    
    # 验证梯度状态保持不变
    assert result.requires_grad == original_requires_grad, \
        "初始化不应该改变张量的梯度状态"
    
    # 验证张量被修改了
    assert not torch.allclose(result, torch.zeros_like(result)), \
        "张量应该被修改"


def test_random_seed_consistency() -> None:
    """验证随机种子的一致性"""
    shape = (3, 4)
    
    # 第一次运行
    set_random_seed(42)
    tensor1 = torch.empty(shape)
    result1 = init.uniform_(tensor1, 0, 1)
    
    # 第二次运行（相同种子）
    set_random_seed(42)
    tensor2 = torch.empty(shape)
    result2 = init.uniform_(tensor2, 0, 1)
    
    # 验证结果相同
    assert torch.allclose(result1, result2), \
        "相同随机种子应该产生相同结果"
    
    # 第三次运行（不同种子）
    set_random_seed(43)
    tensor3 = torch.empty(shape)
    result3 = init.uniform_(tensor3, 0, 1)
    
    # 验证结果不同（大概率）
    assert not torch.allclose(result1, result3), \
        "不同随机种子应该产生不同结果"


if __name__ == "__main__":
    # 简单的手动测试
    print("运行G1组手动测试...")
    
    # 测试 uniform_
    tensor = torch.empty(3, 4)
    result = init.uniform_(tensor, 0, 1)
    print(f"uniform_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 normal_
    tensor = torch.empty(2, 5)
    result = init.normal_(tensor, 0, 1)
    print(f"normal_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 constant_
    tensor = torch.empty(2, 3)
    result = init.constant_(tensor, 5.0)
    print(f"constant_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 ones_
    tensor = torch.empty(2, 3)
    result = init.ones_(tensor)
    print(f"ones_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 zeros_
    tensor = torch.empty(2, 3)
    result = init.zeros_(tensor)
    print(f"zeros_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    print("G1组所有手动测试完成！")
# ==== BLOCK:FOOTER END ====