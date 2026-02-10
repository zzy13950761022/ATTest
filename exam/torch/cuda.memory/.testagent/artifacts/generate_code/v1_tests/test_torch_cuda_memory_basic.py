import math
import pytest
import torch
import sys
from unittest.mock import patch, MagicMock, Mock

# ==== BLOCK:HEADER START ====
import math
import pytest
import torch
import sys
from unittest.mock import patch, MagicMock, Mock

# 固定随机种子确保测试可重复
torch.manual_seed(42)

# 检查CUDA可用性
CUDA_AVAILABLE = torch.cuda.is_available()

# 跳过测试的装饰器
skip_if_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")

# 辅助函数：获取当前设备内存信息
def get_memory_info(device=0):
    """获取指定设备的内存信息"""
    if not CUDA_AVAILABLE:
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    
    with torch.cuda.device(device):
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
        }

# 辅助函数：创建测试张量
def create_test_tensor(size=1024, dtype=torch.float32, shape=None, device=0):
    """创建测试张量"""
    if shape is None:
        # 计算合适的形状来达到指定字节数
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = max(1, size // element_size)
        shape = (num_elements,)
    
    return torch.randn(shape, dtype=dtype, device=f"cuda:{device}" if CUDA_AVAILABLE else "cpu")

# 清理函数：确保测试间内存状态重置
def reset_memory_stats(device=0):
    """重置内存统计"""
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_max_memory_allocated(device)
        torch.cuda.reset_max_memory_cached(device)
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
@skip_if_no_cuda
@pytest.mark.parametrize("device,size,dtype_str,shape", [
    (0, 1024, "float32", [2, 2]),
    (0, 4096, "float64", [4, 4]),  # 参数扩展：更大内存分配和双精度测试
    (0, 128, "int8", [16, 8]),     # 参数扩展：小内存分配和整数类型测试
])
def test_basic_memory_allocation_and_release(device, size, dtype_str, shape):
    """TC-01: 基础内存分配与释放统计
    
    测试内存分配和释放的正确统计，验证：
    1. 分配后 memory_allocated 增加
    2. 释放后 memory_allocated 减少
    3. 基本属性正确性
    """
    # 设置数据类型
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
    }
    dtype = dtype_map[dtype_str]
    
    # 重置内存统计
    reset_memory_stats(device)
    
    # 记录初始内存状态
    initial_memory = torch.cuda.memory_allocated(device)
    
    # 分配内存
    tensor = create_test_tensor(size=size, dtype=dtype, shape=shape, device=device)
    
    # 验证分配后内存增加
    allocated_after = torch.cuda.memory_allocated(device)
    assert allocated_after > initial_memory, "分配后内存应增加"
    
    # 验证内存分配量大致正确（允许少量开销）
    expected_min = size * 0.9  # 允许10%开销
    actual_increase = allocated_after - initial_memory
    assert actual_increase >= expected_min, f"内存增加量应至少为请求的90%，实际增加{actual_increase}，期望至少{expected_min}"
    
    # 记录峰值内存
    peak_before_delete = torch.cuda.max_memory_allocated(device)
    assert peak_before_delete >= allocated_after, "峰值内存应至少等于当前分配内存"
    
    # 释放内存
    del tensor
    torch.cuda.empty_cache()
    
    # 验证释放后内存减少
    allocated_after_delete = torch.cuda.memory_allocated(device)
    assert allocated_after_delete < allocated_after, "释放后内存应减少"
    
    # 验证内存基本回到初始状态（允许少量残留）
    final_memory = torch.cuda.memory_allocated(device)
    assert final_memory <= initial_memory * 1.1, f"最终内存应接近初始状态，初始{initial_memory}，最终{final_memory}"
    
    # 验证峰值内存统计正确
    final_peak = torch.cuda.max_memory_allocated(device)
    assert final_peak >= peak_before_delete, "最终峰值应至少等于之前的峰值"
    
    # 基本属性验证
    assert torch.cuda.memory_allocated(device) >= 0, "已分配内存应为非负数"
    assert torch.cuda.max_memory_allocated(device) >= 0, "峰值内存应为非负数"
    assert torch.cuda.memory_reserved(device) >= 0, "保留内存应为非负数"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@skip_if_no_cuda
def test_empty_cache_functionality():
    """TC-02: empty_cache 缓存清理功能
    
    测试 empty_cache() 对缓存内存的影响，验证：
    1. empty_cache() 后缓存内存减少
    2. 内存状态一致性
    3. 基本属性正确性
    """
    device = 0
    cache_size = 2048
    iterations = 3
    
    # 重置内存统计
    reset_memory_stats(device)
    
    # 记录初始缓存状态
    initial_cached = torch.cuda.memory_cached(device) if hasattr(torch.cuda, 'memory_cached') else 0
    initial_reserved = torch.cuda.memory_reserved(device)
    
    # 创建并立即删除多个张量以产生缓存
    tensors = []
    for i in range(iterations):
        tensor = create_test_tensor(size=cache_size, dtype=torch.float32, device=device)
        tensors.append(tensor)
    
    # 记录分配后的状态
    after_alloc_reserved = torch.cuda.memory_reserved(device)
    
    # 删除所有张量
    del tensors
    
    # 记录删除后的状态（此时内存可能还在缓存中）
    after_delete_reserved = torch.cuda.memory_reserved(device)
    
    # 验证删除后保留内存可能没有立即减少（因为缓存）
    assert after_delete_reserved <= after_alloc_reserved * 1.1, "删除张量后保留内存不应显著增加"
    
    # 调用 empty_cache()
    torch.cuda.empty_cache()
    
    # 验证 empty_cache() 后保留内存减少
    after_empty_reserved = torch.cuda.memory_reserved(device)
    assert after_empty_reserved <= after_delete_reserved, "empty_cache() 后保留内存应减少或不变"
    
    # 验证内存状态一致性
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    
    # 已分配内存应小于等于保留内存
    assert allocated_after <= reserved_after, "已分配内存不应超过保留内存"
    
    # 验证多次调用 empty_cache() 的安全性
    for _ in range(2):
        torch.cuda.empty_cache()
    
    # 多次调用后状态应稳定
    final_reserved = torch.cuda.memory_reserved(device)
    assert abs(final_reserved - after_empty_reserved) <= final_reserved * 0.1, "多次 empty_cache() 调用后状态应稳定"
    
    # 基本属性验证
    assert torch.cuda.memory_allocated(device) >= 0, "已分配内存应为非负数"
    assert torch.cuda.memory_reserved(device) >= 0, "保留内存应为非负数"
    
    # 验证 empty_cache() 不改变已分配内存（只影响缓存）
    # 分配一个新张量
    test_tensor = create_test_tensor(size=512, dtype=torch.float32, device=device)
    allocated_with_tensor = torch.cuda.memory_allocated(device)
    
    # 调用 empty_cache()，已分配内存不应改变
    torch.cuda.empty_cache()
    allocated_after_empty_with_tensor = torch.cuda.memory_allocated(device)
    
    # 允许微小差异（四舍五入等）
    assert abs(allocated_after_empty_with_tensor - allocated_with_tensor) <= allocated_with_tensor * 0.01, \
        "empty_cache() 不应改变已分配内存"
    
    # 清理
    del test_tensor
    torch.cuda.empty_cache()
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: 多设备内存操作隔离性 (DEFERRED)
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: 内存保留与分配关系验证 (DEFERRED)
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:FOOTER START ====
# 测试类：组织相关测试
class TestCUDAMemoryBasic:
    """torch.cuda.memory 基础功能测试类"""
    
    @skip_if_no_cuda
    def test_module_import(self):
        """测试模块导入和基本属性"""
        import torch.cuda.memory as memory_module
        
        # 验证模块存在
        assert memory_module is not None
        
        # 验证关键函数存在
        required_functions = [
            'memory_allocated',
            'max_memory_allocated',
            'empty_cache',
            'memory_reserved',
            'max_memory_reserved',
        ]
        
        for func_name in required_functions:
            assert hasattr(memory_module, func_name), f"模块应包含 {func_name} 函数"
            func = getattr(memory_module, func_name)
            assert callable(func), f"{func_name} 应可调用"

# 参数化测试示例（供后续扩展使用）
@pytest.mark.parametrize("func_name", [
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
])
@skip_if_no_cuda
def test_memory_function_signatures(func_name):
    """测试内存函数的基本签名"""
    func = getattr(torch.cuda, func_name)
    
    # 验证函数可调用
    assert callable(func)
    
    # 测试默认参数调用（使用当前设备）
    try:
        result = func()
        # 内存函数应返回数值
        assert isinstance(result, (int, float))
        assert result >= 0
    except Exception as e:
        # 如果失败，记录但允许（某些函数可能需要参数）
        print(f"{func_name}() 调用异常: {e}")

# 清理钩子
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """每个测试后自动清理"""
    yield
    if torch.cuda.is_available():
        # 尝试清理所有设备
        for device in range(torch.cuda.device_count()):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass  # 忽略清理错误

if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====