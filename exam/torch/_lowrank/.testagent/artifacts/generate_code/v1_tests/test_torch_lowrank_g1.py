import math
import pytest
import torch
from torch._lowrank import svd_lowrank, get_approximate_basis, pca_lowrank

# ==== BLOCK:HEADER START ====
# 测试辅助函数和固定装置
def set_random_seed():
    """设置随机种子确保测试可重复性"""
    torch.manual_seed(42)

def create_test_matrix(shape, dtype=torch.float32, device='cpu', flags=None):
    """创建测试矩阵"""
    if flags is None:
        flags = []
    
    m, n = shape
    # 生成随机矩阵
    A = torch.randn(m, n, dtype=dtype, device=device)
    
    # 根据标志调整矩阵
    if 'full_rank' in flags:
        # 确保满秩
        if m <= n:
            A = torch.eye(m, n, dtype=dtype, device=device)
        else:
            A = torch.eye(m, n, dtype=dtype, device=device)
    elif 'normal' in flags:
        # 已经是正态分布，无需额外处理
        pass
    elif 'random_seed' in flags:
        # 使用特定随机种子
        torch.manual_seed(123)
        A = torch.randn(m, n, dtype=dtype, device=device)
        torch.manual_seed(42)  # 恢复默认种子
    
    return A

def assert_tensor_properties(tensor, expected_shape=None, expected_dtype=None, name=""):
    """断言张量基本属性"""
    assert torch.isfinite(tensor).all(), f"{name} 包含非有限值"
    if expected_shape is not None:
        assert tensor.shape == expected_shape, f"{name} 形状错误: 期望 {expected_shape}, 实际 {tensor.shape}"
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"{name} 数据类型错误: 期望 {expected_dtype}, 实际 {tensor.dtype}"

# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize("dtype,device,shape,q,niter,M,flags", [
    (torch.float32, 'cpu', (5, 3), 2, 2, None, ['normal']),
    # 参数扩展：更大形状测试
    (torch.float64, 'cpu', (8, 5), 3, 2, None, ['larger_shape']),
    # 参数扩展：高矩阵测试
    (torch.float32, 'cpu', (3, 7), 2, 2, None, ['tall_matrix']),
])
def test_svd_lowrank_basic(dtype, device, shape, q, niter, M, flags):
    """测试 svd_lowrank 基本功能"""
    # 设置随机种子
    set_random_seed()
    
    # 创建测试矩阵
    A = create_test_matrix(shape, dtype=dtype, device=device, flags=flags)
    
    # 调用目标函数
    U, S, V = svd_lowrank(A, q=q, niter=niter, M=M)
    
    # weak 断言：基本属性
    m, n = shape
    assert_tensor_properties(U, expected_shape=(m, q), expected_dtype=dtype, name="U")
    assert_tensor_properties(S, expected_shape=(q,), expected_dtype=dtype, name="S")
    assert_tensor_properties(V, expected_shape=(n, q), expected_dtype=dtype, name="V")
    
    # weak 断言：基本属性检查
    # 1. 奇异值应为非负
    assert (S >= 0).all(), "奇异值应非负"
    
    # 2. 奇异值应为降序排列（近似）
    if q > 1:
        # 检查是否大致降序（允许小的数值误差）
        diff = S[:-1] - S[1:]
        assert (diff >= -1e-6).all(), "奇异值应大致降序排列"
    
    # 3. 重构误差检查（弱断言版本）
    # 计算低秩近似
    A_approx = U @ torch.diag(S) @ V.T
    
    # 检查重构矩阵形状
    assert A_approx.shape == A.shape, f"重构矩阵形状错误: 期望 {A.shape}, 实际 {A_approx.shape}"
    
    # 检查重构误差数量级（弱断言，只检查不是特别大）
    error = torch.norm(A - A_approx)
    norm_A = torch.norm(A)
    if norm_A > 0:
        rel_error = error / norm_A
        # 弱断言：相对误差应小于 0.1（宽松条件）
        assert rel_error < 0.1, f"重构相对误差过大: {rel_error.item()}"
    
    # 4. 正交性检查（弱断言版本）
    # U 的列应近似正交
    UUT = U.T @ U
    eye_U = torch.eye(q, dtype=dtype, device=device)
    orth_error_U = torch.norm(UUT - eye_U)
    # 弱断言：正交性误差应小于 0.1
    assert orth_error_U < 0.1, f"U 正交性误差过大: {orth_error_U.item()}"
    
    # V 的列应近似正交
    VVT = V.T @ V
    eye_V = torch.eye(q, dtype=dtype, device=device)
    orth_error_V = torch.norm(VVT - eye_V)
    # 弱断言：正交性误差应小于 0.1
    assert orth_error_V < 0.1, f"V 正交性误差过大: {orth_error_V.item()}"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("dtype,device,shape,q,niter,M,flags", [
    (torch.float64, 'cpu', (4, 4), 4, 0, None, ['full_rank']),
    # 参数扩展：最小形状测试
    (torch.float32, 'cpu', (1, 1), 1, 0, None, ['minimal']),
])
def test_svd_lowrank_boundary(dtype, device, shape, q, niter, M, flags):
    """测试 svd_lowrank 边界条件（满秩情况）"""
    # 设置随机种子
    set_random_seed()
    
    # 创建测试矩阵（满秩矩阵）
    A = create_test_matrix(shape, dtype=dtype, device=device, flags=flags)
    
    # 调用目标函数
    U, S, V = svd_lowrank(A, q=q, niter=niter, M=M)
    
    # weak 断言：基本属性
    m, n = shape
    assert_tensor_properties(U, expected_shape=(m, q), expected_dtype=dtype, name="U")
    assert_tensor_properties(S, expected_shape=(q,), expected_dtype=dtype, name="S")
    assert_tensor_properties(V, expected_shape=(n, q), expected_dtype=dtype, name="V")
    
    # weak 断言：基本属性检查
    # 1. 奇异值应为非负
    assert (S >= 0).all(), "奇异值应非负"
    
    # 2. 对于满秩矩阵，所有奇异值应大于 0（近似）
    # 注意：由于数值误差，可能有一些很小的奇异值
    assert (S > -1e-10).all(), "满秩矩阵的奇异值应大于 0"
    
    # 3. 重构误差检查（弱断言版本）
    # 计算低秩近似（这里 q = min(m,n)，所以是完整重构）
    A_approx = U @ torch.diag(S) @ V.T
    
    # 检查重构矩阵形状
    assert A_approx.shape == A.shape, f"重构矩阵形状错误: 期望 {A.shape}, 实际 {A_approx.shape}"
    
    # 对于满秩情况，重构应更精确
    error = torch.norm(A - A_approx)
    norm_A = torch.norm(A)
    if norm_A > 0:
        rel_error = error / norm_A
        # 弱断言：相对误差应小于 0.05（比基本测试更严格）
        assert rel_error < 0.05, f"满秩重构相对误差过大: {rel_error.item()}"
    
    # 4. 正交性检查（弱断言版本）
    # U 的列应近似正交
    UUT = U.T @ U
    eye_U = torch.eye(q, dtype=dtype, device=device)
    orth_error_U = torch.norm(UUT - eye_U)
    # 弱断言：正交性误差应小于 0.05
    assert orth_error_U < 0.05, f"U 正交性误差过大: {orth_error_U.item()}"
    
    # V 的列应近似正交
    VVT = V.T @ V
    eye_V = torch.eye(q, dtype=dtype, device=device)
    orth_error_V = torch.norm(VVT - eye_V)
    # 弱断言：正交性误差应小于 0.05
    assert orth_error_V < 0.05, f"V 正交性误差过大: {orth_error_V.item()}"
    
    # 5. 检查 niter=0 的情况
    # niter=0 表示没有子空间迭代，算法应仍然工作
    if 'minimal' not in flags:  # 最小形状测试可能不适用此检查
        assert niter == 0, "此测试专门验证 niter=0 的边界情况"
    
    # 6. 检查 q = min(m,n) 的情况
    if 'minimal' not in flags:  # 最小形状测试可能不适用此检查
        assert q == min(m, n), "此测试专门验证 q = min(m,n) 的边界情况"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_04 START ====
# svd_lowrank 随机性控制（延后）
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:FOOTER START ====
def test_svd_lowrank_invalid_q():
    """测试无效 q 参数"""
    set_random_seed()
    A = torch.randn(5, 3, dtype=torch.float32)
    
    # q > min(m,n) 应引发 ValueError
    with pytest.raises(ValueError, match="q must be satisfying"):
        svd_lowrank(A, q=10)  # q=10 > min(5,3)=3
    
    # q < 0 应引发 ValueError
    with pytest.raises(ValueError, match="q must be satisfying"):
        svd_lowrank(A, q=-1)

def test_svd_lowrank_invalid_niter():
    """测试无效 niter 参数"""
    set_random_seed()
    A = torch.randn(5, 3, dtype=torch.float32)
    
    # niter < 0 应引发 ValueError
    with pytest.raises(ValueError, match="niter must be non-negative"):
        svd_lowrank(A, niter=-1)

def test_zero_q():
    """测试 q=0 的边界情况"""
    set_random_seed()
    A = torch.randn(5, 3, dtype=torch.float32)
    
    # svd_lowrank with q=0
    U, S, V = svd_lowrank(A, q=0)
    assert U.shape == (5, 0)
    assert S.shape == (0,)
    assert V.shape == (3, 0)
    
    # 检查空张量的属性
    assert U.dtype == torch.float32
    assert S.dtype == torch.float32
    assert V.dtype == torch.float32
# ==== BLOCK:FOOTER END ====