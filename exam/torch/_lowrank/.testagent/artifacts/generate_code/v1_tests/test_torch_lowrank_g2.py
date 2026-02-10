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
    if 'normal' in flags:
        # 已经是正态分布，无需额外处理
        pass
    elif 'centered' in flags:
        # 创建中心化测试数据
        # 添加一些偏移，然后测试中心化效果
        offset = torch.randn(m, 1, dtype=dtype, device=device) * 5.0
        A = A + offset @ torch.ones(1, n, dtype=dtype, device=device)
    elif 'sparse' in flags:
        # 创建稀疏矩阵（模拟稀疏性）
        # 实际上我们创建密集矩阵，但测试稀疏兼容性
        # 真正的稀疏矩阵测试需要特殊处理
        pass
    
    return A

def assert_tensor_properties(tensor, expected_shape=None, expected_dtype=None, name=""):
    """断言张量基本属性"""
    assert torch.isfinite(tensor).all(), f"{name} 包含非有限值"
    if expected_shape is not None:
        assert tensor.shape == expected_shape, f"{name} 形状错误: 期望 {expected_shape}, 实际 {tensor.shape}"
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"{name} 数据类型错误: 期望 {expected_dtype}, 实际 {tensor.dtype}"

# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize("dtype,device,shape,q,niter,M,flags", [
    (torch.float32, 'cpu', (6, 4), 3, 2, None, ['normal']),
    # 参数扩展：不同迭代次数
    (torch.float64, 'cpu', (5, 3), 2, 1, None, ['different_niter']),
])
def test_get_approximate_basis_basic(dtype, device, shape, q, niter, M, flags):
    """测试 get_approximate_basis 基本功能"""
    # 设置随机种子
    set_random_seed()
    
    # 创建测试矩阵
    A = create_test_matrix(shape, dtype=dtype, device=device, flags=flags)
    
    # 调用目标函数
    Q = get_approximate_basis(A, q=q, niter=niter, M=M)
    
    # weak 断言：基本属性
    m, n = shape
    assert_tensor_properties(Q, expected_shape=(m, q), expected_dtype=dtype, name="Q")
    
    # weak 断言：基本属性检查
    # 1. 正交性检查（弱断言版本）
    QTQ = Q.T @ Q
    eye = torch.eye(q, dtype=dtype, device=device)
    orth_error = torch.norm(QTQ - eye)
    # 弱断言：正交性误差应小于 0.1
    assert orth_error < 0.1, f"Q 正交性误差过大: {orth_error.item()}"
    
    # 2. 检查 Q 的列范数
    col_norms = torch.norm(Q, dim=0)
    # 列范数应接近 1（正交归一化）
    norm_error = torch.norm(col_norms - 1.0)
    # 弱断言：范数误差应小于 0.1
    assert norm_error < 0.1, f"Q 列范数误差过大: {norm_error.item()}"
    
    # 3. 近似质量检查（弱断言）
    # 计算 Q Q^T A 作为 A 的近似
    A_approx = Q @ Q.T @ A
    
    # 检查近似矩阵形状
    assert A_approx.shape == A.shape, f"近似矩阵形状错误: 期望 {A.shape}, 实际 {A_approx.shape}"
    
    # 检查近似误差数量级（弱断言）
    error = torch.norm(A - A_approx)
    norm_A = torch.norm(A)
    if norm_A > 0:
        rel_error = error / norm_A
        # 弱断言：相对误差应小于 0.5（对于随机投影算法，这个条件比较宽松）
        assert rel_error < 0.5, f"近似相对误差过大: {rel_error.item()}"
    
    # 4. 检查 niter 参数
    if 'different_niter' in flags:
        assert niter == 1, "此测试验证 niter=1 的情况"
    else:
        assert niter == 2, "此测试使用默认 niter=2"
    
    # 5. 检查 M=None 的情况
    assert M is None, "此测试验证 M=None 的情况"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_05 START ====
# pca_lowrank 中心化测试（延后）
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# pca_lowrank 稀疏矩阵支持（延后）
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:FOOTER START ====
def test_get_approximate_basis_invalid_q():
    """测试 get_approximate_basis 无效 q 参数"""
    set_random_seed()
    A = torch.randn(6, 4, dtype=torch.float32)
    
    # q > min(m,n) 应引发 ValueError
    with pytest.raises(ValueError, match="q must be satisfying"):
        get_approximate_basis(A, q=10)  # q=10 > min(6,4)=4
    
    # q < 0 应引发 ValueError
    with pytest.raises(ValueError, match="q must be satisfying"):
        get_approximate_basis(A, q=-1)

def test_get_approximate_basis_invalid_niter():
    """测试 get_approximate_basis 无效 niter 参数"""
    set_random_seed()
    A = torch.randn(6, 4, dtype=torch.float32)
    
    # niter < 0 应引发 ValueError
    with pytest.raises(ValueError, match="niter must be non-negative"):
        get_approximate_basis(A, niter=-1)

def test_pca_lowrank_basic():
    """测试 pca_lowrank 基本功能（简单验证）"""
    set_random_seed()
    A = torch.randn(5, 3, dtype=torch.float32)
    
    # 测试默认参数
    U, S, V = pca_lowrank(A, q=2)
    
    # 基本形状检查
    assert U.shape == (5, 2)
    assert S.shape == (2,)
    assert V.shape == (3, 2)
    
    # 数据类型检查
    assert U.dtype == torch.float32
    assert S.dtype == torch.float32
    assert V.dtype == torch.float32
    
    # 奇异值非负
    assert (S >= 0).all()

def test_get_approximate_basis_zero_q():
    """测试 get_approximate_basis q=0 的边界情况"""
    set_random_seed()
    A = torch.randn(5, 3, dtype=torch.float32)
    
    # get_approximate_basis with q=0
    Q = get_approximate_basis(A, q=0)
    assert Q.shape == (5, 0)
    assert Q.dtype == torch.float32
# ==== BLOCK:FOOTER END ====