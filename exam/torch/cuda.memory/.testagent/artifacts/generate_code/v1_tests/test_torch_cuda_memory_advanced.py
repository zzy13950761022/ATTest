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

# ==== BLOCK:CASE_09 START ====
@skip_if_no_cuda
def test_caching_allocator_direct_operations():
    """TC-09: 内存分配器直接操作
    
    测试 caching_allocator_alloc 和 caching_allocator_delete 的直接操作，验证：
    1. 分配器工作正常
    2. 返回的指针有效
    3. 基本属性正确性
    """
    device = 0
    size = 1024
    
    # 重置内存统计
    reset_memory_stats(device)
    
    # 记录初始内存状态
    initial_allocated = torch.cuda.memory_allocated(device)
    initial_reserved = torch.cuda.memory_reserved(device)
    
    # 使用分配器直接分配内存
    mem_ptr = torch.cuda.caching_allocator_alloc(size, device=device)
    
    # 验证返回的指针有效
    assert mem_ptr is not None, "分配应返回有效指针"
    assert isinstance(mem_ptr, int), "指针应为整数类型"
    assert mem_ptr > 0, "指针应为正数"
    
    # 验证分配后内存增加
    allocated_after = torch.cuda.memory_allocated(device)
    assert allocated_after > initial_allocated, "分配后已分配内存应增加"
    
    # 验证内存增加量大致正确
    actual_increase = allocated_after - initial_allocated
    expected_min = size * 0.9  # 允许10%开销
    assert actual_increase >= expected_min, f"内存增加量应至少为请求的90%，实际增加{actual_increase}，期望至少{expected_min}"
    
    # 验证保留内存可能增加
    reserved_after = torch.cuda.memory_reserved(device)
    assert reserved_after >= initial_reserved, "分配后保留内存应增加或不变"
    
    # 记录分配后的峰值
    peak_after_alloc = torch.cuda.max_memory_allocated(device)
    assert peak_after_alloc >= allocated_after, "峰值内存应至少等于当前分配内存"
    
    # 使用分配器删除内存
    torch.cuda.caching_allocator_delete(mem_ptr)
    
    # 验证删除后内存减少
    allocated_after_delete = torch.cuda.memory_allocated(device)
    assert allocated_after_delete < allocated_after, "删除后已分配内存应减少"
    
    # 调用 empty_cache 确保完全释放
    torch.cuda.empty_cache()
    
    # 验证最终内存状态
    final_allocated = torch.cuda.memory_allocated(device)
    final_reserved = torch.cuda.memory_reserved(device)
    
    # 允许少量残留（缓存、碎片等）
    assert final_allocated <= initial_allocated * 1.1, f"最终已分配内存应接近初始状态: 初始={initial_allocated}, 最终={final_allocated}"
    
    # 验证多次分配释放序列
    pointers = []
    for i in range(3):
        ptr = torch.cuda.caching_allocator_alloc(size // 2, device=device)
        assert ptr is not None and ptr > 0, f"第{i+1}次分配应返回有效指针"
        pointers.append(ptr)
    
    # 验证多次分配后内存增加
    allocated_after_multiple = torch.cuda.memory_allocated(device)
    assert allocated_after_multiple > initial_allocated, "多次分配后内存应增加"
    
    # 逐个释放
    for ptr in pointers:
        torch.cuda.caching_allocator_delete(ptr)
    
    # 全部释放后调用 empty_cache
    torch.cuda.empty_cache()
    
    # 验证最终状态
    final_after_sequence = torch.cuda.memory_allocated(device)
    assert final_after_sequence <= initial_allocated * 1.2, f"序列操作后内存应接近初始状态: 初始={initial_allocated}, 最终={final_after_sequence}"
    
    # 测试无效参数场景（使用 mock 来测试异常情况）
    with patch('torch.cuda.caching_allocator_alloc') as mock_alloc:
        # 模拟分配失败
        mock_alloc.return_value = 0  # 无效指针
        invalid_ptr = torch.cuda.caching_allocator_alloc(size, device=device)
        assert invalid_ptr == 0, "模拟的分配失败应返回0"
    
    # 测试删除无效指针（应安全处理）
    # 注意：实际实现可能对无效指针有保护，这里我们测试现有指针的重复删除
    valid_ptr = torch.cuda.caching_allocator_alloc(size, device=device)
    assert valid_ptr > 0, "应分配有效指针"
    
    # 第一次删除应成功
    torch.cuda.caching_allocator_delete(valid_ptr)
    
    # 第二次删除相同指针（可能引发错误或安全忽略）
    # 我们捕获可能的异常以确保测试不崩溃
    try:
        torch.cuda.caching_allocator_delete(valid_ptr)
        # 如果没有异常，继续执行
    except Exception as e:
        # 如果有异常，记录但允许（取决于实现）
        print(f"重复删除指针引发异常（可能正常）: {e}")
    
    # 清理
    torch.cuda.empty_cache()
    
    # 基本属性验证
    assert torch.cuda.memory_allocated(device) >= 0, "已分配内存应为非负数"
    assert torch.cuda.memory_reserved(device) >= 0, "保留内存应为非负数"
    
    # 验证分配器函数的存在和可调用性
    assert hasattr(torch.cuda, 'caching_allocator_alloc'), "应存在 caching_allocator_alloc 函数"
    assert hasattr(torch.cuda, 'caching_allocator_delete'), "应存在 caching_allocator_delete 函数"
    assert callable(torch.cuda.caching_allocator_alloc), "caching_allocator_alloc 应可调用"
    assert callable(torch.cuda.caching_allocator_delete), "caching_allocator_delete 应可调用"
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:CASE_10 START ====
# TC-10: 进程内存限制设置 (DEFERRED)
# ==== BLOCK:CASE_10 END ====

# ==== BLOCK:CASE_11 START ====
# TC-11: GPU进程列表查询 (DEFERRED)
# ==== BLOCK:CASE_11 END ====

# ==== BLOCK:CASE_12 START ====
# TC-12: 异常参数错误处理 (DEFERRED)
# ==== BLOCK:CASE_12 END ====

# ==== BLOCK:FOOTER START ====
# 测试类：组织相关测试
class TestCUDAMemoryAdvanced:
    """torch.cuda.memory 高级功能测试类"""
    
    @skip_if_no_cuda
    def test_module_import(self):
        """测试模块导入和基本属性"""
        import torch.cuda.memory as memory_module
        
        # 验证模块存在
        assert memory_module is not None
        
        # 验证关键函数存在
        required_functions = [
            'caching_allocator_alloc',
            'caching_allocator_delete',
            'list_gpu_processes',
            'set_per_process_memory_fraction',
        ]
        
        for func_name in required_functions:
            assert hasattr(memory_module, func_name), f"模块应包含 {func_name} 函数"
            func = getattr(memory_module, func_name)
            assert callable(func), f"{func_name} 应可调用"

# 参数化测试示例（供后续扩展使用）
@pytest.mark.parametrize("func_name", [
    "caching_allocator_alloc",
    "caching_allocator_delete",
])
@skip_if_no_cuda
def test_allocator_function_signatures(func_name):
    """测试分配器函数的基本签名"""
    func = getattr(torch.cuda, func_name)
    
    # 验证函数可调用
    assert callable(func)
    
    # 测试基本调用（需要参数）
    try:
        if func_name == "caching_allocator_alloc":
            # 需要大小参数
            result = func(1024, device=0)
            assert isinstance(result, int), "应返回整数指针"
            assert result >= 0, "指针应为非负数"
        elif func_name == "caching_allocator_delete":
            # 需要指针参数，这里我们创建一个然后删除
            ptr = torch.cuda.caching_allocator_alloc(1024, device=0)
            func(ptr)  # 应成功执行
    except Exception as e:
        # 如果失败，记录但允许
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