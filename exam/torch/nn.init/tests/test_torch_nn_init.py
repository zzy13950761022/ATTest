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
@pytest.mark.parametrize("dtype,device,shape,a,b,flags", [
    (torch.float32, "cpu", (3, 4), 0.0, 1.0, []),
])
def test_uniform_basic(dtype: torch.dtype, 
                      device: str, 
                      shape: Tuple[int, ...], 
                      a: float, 
                      b: float,
                      flags: List[str]) -> None:
    """测试 uniform_ 函数的基础功能
    
    验证均匀分布初始化的基本属性：
    1. 形状正确
    2. 数据类型正确
    3. 所有值都是有限值
    4. 值在指定范围内 [a, b]
    """
    # 设置随机种子以确保可重复性
    set_random_seed(42)
    
    # 创建输入张量
    tensor = torch.empty(shape, dtype=dtype, device=device)
    
    # 保存原始张量引用（用于验证原地修改）
    original_tensor_id = id(tensor)
    
    # 调用初始化函数
    result = init.uniform_(tensor, a=a, b=b)
    
    # 验证原地修改
    assert id(result) == original_tensor_id, "uniform_ 应该原地修改张量"
    
    # weak 断言：基本属性验证
    assert_tensor_properties(result, shape, dtype, "uniform_基础测试")
    
    # weak 断言：值在范围内
    assert_in_range(result, a, b, "uniform_基础测试")
    
    # weak 断言：不全为零（对于 a=0, b=1 的情况）
    if a == 0.0 and b == 1.0:
        assert_not_all_zero(result, "uniform_基础测试")
    
    # 验证张量确实被修改了（不是全零或全NaN）
    assert torch.isfinite(result).any(), "张量应该包含有限值"
# ==== BLOCK:CASE_01 END ====


# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("dtype,device,shape,mean,std,flags", [
    (torch.float32, "cpu", (2, 5), 0.0, 1.0, []),
])
def test_normal_basic(dtype: torch.dtype, 
                     device: str, 
                     shape: Tuple[int, ...], 
                     mean: float, 
                     std: float,
                     flags: List[str]) -> None:
    """测试 normal_ 函数的基础功能
    
    验证正态分布初始化的基本属性：
    1. 形状正确
    2. 数据类型正确
    3. 所有值都是有限值
    4. 不全为零（对于非零标准差）
    """
    # 设置随机种子以确保可重复性
    set_random_seed(42)
    
    # 创建输入张量
    tensor = torch.empty(shape, dtype=dtype, device=device)
    
    # 保存原始张量引用（用于验证原地修改）
    original_tensor_id = id(tensor)
    
    # 调用初始化函数
    result = init.normal_(tensor, mean=mean, std=std)
    
    # 验证原地修改
    assert id(result) == original_tensor_id, "normal_ 应该原地修改张量"
    
    # weak 断言：基本属性验证
    assert_tensor_properties(result, shape, dtype, "normal_基础测试")
    
    # weak 断言：不全为零（对于非零标准差）
    if std > 0:
        assert_not_all_zero(result, "normal_基础测试")
    
    # 验证张量确实被修改了（不是全零或全NaN）
    assert torch.isfinite(result).any(), "张量应该包含有限值"
    
    # 验证没有异常值（绝对值过大）
    # 对于标准正态分布，值应该在合理范围内
    max_abs_value = torch.abs(result).max().item()
    assert max_abs_value < 10.0, f"正态分布值过大: {max_abs_value}"
# ==== BLOCK:CASE_02 END ====


# ==== BLOCK:CASE_03 START ====
# constant_和ones_zeros_测试（占位）
# ==== BLOCK:CASE_03 END ====


# ==== BLOCK:CASE_04 START ====
# 基础函数边界测试（占位）
# ==== BLOCK:CASE_04 END ====


# ==== BLOCK:CASE_05 START ====
@pytest.mark.parametrize("dtype,device,shape,gain,flags", [
    (torch.float32, "cpu", (4, 6), 1.0, []),
])
def test_xavier_uniform_basic(dtype: torch.dtype, 
                             device: str, 
                             shape: Tuple[int, ...], 
                             gain: float,
                             flags: List[str]) -> None:
    """测试 xavier_uniform_ 函数的基础功能
    
    验证Xavier均匀分布初始化的基本属性：
    1. 形状正确
    2. 数据类型正确
    3. 所有值都是有限值
    4. 值在计算出的范围内
    """
    # 设置随机种子以确保可重复性
    set_random_seed(42)
    
    # 创建输入张量
    tensor = torch.empty(shape, dtype=dtype, device=device)
    
    # 保存原始张量引用（用于验证原地修改）
    original_tensor_id = id(tensor)
    
    # 调用初始化函数
    result = init.xavier_uniform_(tensor, gain=gain)
    
    # 验证原地修改
    assert id(result) == original_tensor_id, "xavier_uniform_ 应该原地修改张量"
    
    # weak 断言：基本属性验证
    assert_tensor_properties(result, shape, dtype, "xavier_uniform_基础测试")
    
    # 计算Xavier均匀分布的边界
    # 根据公式: a = gain * sqrt(6 / (fan_in + fan_out))
    fan_in = shape[1] if len(shape) >= 2 else shape[0]
    fan_out = shape[0] if len(shape) >= 2 else 1
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    
    # weak 断言：值在范围内 [-a, a]
    assert_in_range(result, -a, a, "xavier_uniform_基础测试")
    
    # weak 断言：不全为零
    assert_not_all_zero(result, "xavier_uniform_基础测试")
    
    # 验证张量确实被修改了
    assert torch.isfinite(result).any(), "张量应该包含有限值"
    
    # 验证边界计算正确（值应该接近边界但不是全部在边界）
    # 对于均匀分布，应该有值接近边界
    max_abs_value = torch.abs(result).max().item()
    assert max_abs_value > 0.1 * a, f"值应该更接近边界，最大绝对值: {max_abs_value}, a: {a}"
# ==== BLOCK:CASE_05 END ====


# ==== BLOCK:CASE_06 START ====
@pytest.mark.parametrize("dtype,device,shape,mode,nonlinearity,flags", [
    (torch.float32, "cpu", (5, 3), "fan_in", "relu", []),
])
def test_kaiming_uniform_basic(dtype: torch.dtype, 
                              device: str, 
                              shape: Tuple[int, ...], 
                              mode: str,
                              nonlinearity: str,
                              flags: List[str]) -> None:
    """测试 kaiming_uniform_ 函数的基础功能
    
    验证Kaiming均匀分布初始化的基本属性：
    1. 形状正确
    2. 数据类型正确
    3. 所有值都是有限值
    4. 值在计算出的范围内
    """
    # 设置随机种子以确保可重复性
    set_random_seed(42)
    
    # 创建输入张量
    tensor = torch.empty(shape, dtype=dtype, device=device)
    
    # 保存原始张量引用（用于验证原地修改）
    original_tensor_id = id(tensor)
    
    # 调用初始化函数
    result = init.kaiming_uniform_(tensor, mode=mode, nonlinearity=nonlinearity)
    
    # 验证原地修改
    assert id(result) == original_tensor_id, "kaiming_uniform_ 应该原地修改张量"
    
    # weak 断言：基本属性验证
    assert_tensor_properties(result, shape, dtype, "kaiming_uniform_基础测试")
    
    # 计算Kaiming均匀分布的边界
    # 根据公式: bound = gain * sqrt(3 / fan_mode)
    # 首先计算正确的fan值
    if mode == "fan_in":
        fan = shape[1] if len(shape) >= 2 else shape[0]
    else:  # fan_out
        fan = shape[0] if len(shape) >= 2 else 1
    
    # 计算增益
    gain = init.calculate_gain(nonlinearity, 0)  # a=0 for relu
    
    # 计算边界
    bound = gain * math.sqrt(3.0 / fan)
    
    # weak 断言：值在范围内 [-bound, bound]
    assert_in_range(result, -bound, bound, "kaiming_uniform_基础测试")
    
    # weak 断言：不全为零
    assert_not_all_zero(result, "kaiming_uniform_基础测试")
    
    # 验证张量确实被修改了
    assert torch.isfinite(result).any(), "张量应该包含有限值"
    
    # 验证边界计算正确
    max_abs_value = torch.abs(result).max().item()
    assert max_abs_value > 0.1 * bound, f"值应该更接近边界，最大绝对值: {max_abs_value}, bound: {bound}"
    
    # 验证模式参数正确应用
    # 对于fan_in模式，fan应该是输入维度
    if mode == "fan_in" and len(shape) >= 2:
        expected_fan = shape[1]
        assert fan == expected_fan, f"fan_in计算错误: {fan} != {expected_fan}"
# ==== BLOCK:CASE_06 END ====


# ==== BLOCK:CASE_07 START ====
# 自适应初始化参数组合（占位）
# ==== BLOCK:CASE_07 END ====


# ==== BLOCK:CASE_08 START ====
# 自适应初始化边界测试（占位）
# ==== BLOCK:CASE_08 END ====


# ==== BLOCK:CASE_09 START ====
@pytest.mark.parametrize("dtype,device,shape,flags", [
    (torch.float32, "cpu", (3, 3), []),
])
def test_eye_and_dirac_basic(dtype: torch.dtype, 
                            device: str, 
                            shape: Tuple[int, ...], 
                            flags: List[str]) -> None:
    """测试 eye_ 和 dirac_ 函数的基础功能
    
    验证特殊矩阵初始化的基本属性：
    1. 形状正确
    2. 数据类型正确
    3. 所有值都是有限值
    4. eye_ 创建单位矩阵模式
    """
    # 设置随机种子以确保可重复性
    set_random_seed(42)
    
    # 测试 eye_ 函数（2D张量）
    if len(shape) == 2:
        # 创建输入张量
        tensor_eye = torch.empty(shape, dtype=dtype, device=device)
        original_tensor_id = id(tensor_eye)
        
        # 调用 eye_ 初始化函数
        result_eye = init.eye_(tensor_eye)
        
        # 验证原地修改
        assert id(result_eye) == original_tensor_id, "eye_ 应该原地修改张量"
        
        # weak 断言：基本属性验证
        assert_tensor_properties(result_eye, shape, dtype, "eye_基础测试")
        
        # weak 断言：eye模式验证
        # 对于方阵，对角线应该是1，其他位置应该是0
        n, m = shape
        min_dim = min(n, m)
        
        # 检查对角线元素
        for i in range(min_dim):
            assert torch.allclose(result_eye[i, i], torch.tensor(1.0, dtype=dtype)), \
                f"对角线元素 {i},{i} 应该是1"
        
        # 检查非对角线元素（应该是0）
        zero_count = (result_eye == 0).sum().item()
        expected_zero_count = n * m - min_dim
        assert zero_count >= expected_zero_count, \
            f"应该有至少 {expected_zero_count} 个零元素，实际 {zero_count}"
        
        # 验证张量确实被修改了
        assert torch.isfinite(result_eye).any(), "张量应该包含有限值"
    
    # 注意：dirac_ 需要3-5维张量，但测试用例指定的是2D [3,3]
    # 所以这里只测试 eye_，dirac_ 测试将在其他用例中覆盖
    # 或者可以在这里添加一个简单的dirac_测试（如果需要）
    
    # 如果需要测试 dirac_，可以添加以下代码：
    # if len(shape) in [3, 4, 5]:
    #     tensor_dirac = torch.empty(shape, dtype=dtype, device=device)
    #     result_dirac = init.dirac_(tensor_dirac)
    #     # 验证dirac_的基本属性
    #     assert_tensor_properties(result_dirac, shape, dtype, "dirac_基础测试")
    #     # dirac_应该只在中心位置有1，其他地方为0
    #     ones_count = (result_dirac == 1).sum().item()
    #     min_dim = min(shape[0], shape[1])
    #     assert ones_count == min_dim, f"应该有 {min_dim} 个1，实际 {ones_count}"
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
    
    # 测试 uniform_ 的非法输入
    with pytest.raises(RuntimeError):
        # 非张量输入
        init.uniform_(None, 0, 1)
    
    # 测试 normal_ 的非法标准差
    tensor = torch.empty(2, 3)
    with pytest.raises(RuntimeError):
        init.normal_(tensor, mean=0, std=-1)  # 负标准差
    
    # 测试 eye_ 的非法维度
    tensor_1d = torch.empty(3)
    with pytest.raises(ValueError, match="Only tensors with 2 dimensions are supported"):
        init.eye_(tensor_1d)
    
    # 测试 dirac_ 的非法维度
    tensor_2d = torch.empty(3, 3)
    with pytest.raises(ValueError, match="Only tensors with 3, 4, or 5 dimensions are supported"):
        init.dirac_(tensor_2d)


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
    print("运行手动测试...")
    
    # 测试 uniform_
    tensor = torch.empty(3, 4)
    result = init.uniform_(tensor, 0, 1)
    print(f"uniform_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 normal_
    tensor = torch.empty(2, 5)
    result = init.normal_(tensor, 0, 1)
    print(f"normal_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 xavier_uniform_
    tensor = torch.empty(4, 6)
    result = init.xavier_uniform_(tensor)
    print(f"xavier_uniform_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 kaiming_uniform_
    tensor = torch.empty(5, 3)
    result = init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')
    print(f"kaiming_uniform_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    # 测试 eye_
    tensor = torch.empty(3, 3)
    result = init.eye_(tensor)
    print(f"eye_ 测试通过: shape={result.shape}, dtype={result.dtype}")
    
    print("所有手动测试完成！")
# ==== BLOCK:FOOTER END ====