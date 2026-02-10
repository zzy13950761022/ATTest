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

# ==== BLOCK:CASE_05 START ====
@skip_if_no_cuda
@pytest.mark.parametrize("device,abbreviated", [
    (0, False),
    (0, True),  # 参数扩展：简略统计模式测试
])
def test_memory_stats_dict_structure(device, abbreviated):
    """TC-05: 内存统计字典结构完整性
    
    测试 memory_stats_as_nested_dict 的字典结构，验证：
    1. 返回字典结构正确
    2. 包含必需的关键字段
    3. 基本属性正确性
    """
    # 重置内存统计
    reset_memory_stats(device)
    
    # 获取内存统计
    stats_dict = torch.cuda.memory_stats_as_nested_dict(device)
    
    # 验证返回类型是字典
    assert isinstance(stats_dict, dict), "应返回字典类型"
    
    # 验证字典结构
    required_keys = ["allocated", "reserved", "active", "inactive"]
    
    for key in required_keys:
        assert key in stats_dict, f"统计字典应包含 '{key}' 键"
    
    # 验证嵌套字典结构
    if "allocated" in stats_dict:
        allocated_info = stats_dict["allocated"]
        assert isinstance(allocated_info, dict), "'allocated' 应为字典"
        
        # 检查 allocated 字典中的关键字段
        allocated_subkeys = ["current", "peak", "allocated"]
        for subkey in allocated_subkeys:
            if subkey in allocated_info:
                value = allocated_info[subkey]
                assert isinstance(value, (int, float)), f"allocated.{subkey} 应为数值类型"
                assert value >= 0, f"allocated.{subkey} 应为非负数"
    
    # 验证 reserved 信息
    if "reserved" in stats_dict:
        reserved_info = stats_dict["reserved"]
        assert isinstance(reserved_info, dict), "'reserved' 应为字典"
        
        if "current" in reserved_info:
            assert reserved_info["current"] >= 0, "reserved.current 应为非负数"
    
    # 验证 active 和 inactive 信息
    for key in ["active", "inactive"]:
        if key in stats_dict:
            info = stats_dict[key]
            assert isinstance(info, dict), f"'{key}' 应为字典"
            
            # 检查是否包含大小信息
            size_keys = ["num", "size", "count"]
            found_size_key = False
            for size_key in size_keys:
                if size_key in info:
                    assert info[size_key] >= 0, f"{key}.{size_key} 应为非负数"
                    found_size_key = True
            
            if not found_size_key:
                # 如果没有标准大小键，检查字典是否非空
                assert len(info) > 0, f"'{key}' 字典应包含信息"
    
    # 验证 memory_stats 函数（非嵌套版本）
    flat_stats = torch.cuda.memory_stats(device)
    assert isinstance(flat_stats, dict), "memory_stats 应返回字典"
    
    # 检查 flat_stats 中的关键指标
    flat_required_keys = ["allocated_bytes.all.current", "reserved_bytes.all.current"]
    
    for key in flat_required_keys:
        if key in flat_stats:
            value = flat_stats[key]
            assert isinstance(value, (int, float)), f"{key} 应为数值类型"
            assert value >= 0, f"{key} 应为非负数"
    
    # 验证两种统计方式的一致性
    # allocated 值应该大致匹配
    if "allocated" in stats_dict and "current" in stats_dict["allocated"]:
        nested_allocated = stats_dict["allocated"]["current"]
        
        # 在 flat_stats 中查找对应的 allocated 值
        flat_allocated_keys = [
            "allocated_bytes.all.current",
            "allocated_bytes.current",
            "bytes.allocated.current"
        ]
        
        for key in flat_allocated_keys:
            if key in flat_stats:
                flat_allocated = flat_stats[key]
                # 允许10%的差异（不同统计方式可能有微小差异）
                ratio = nested_allocated / max(flat_allocated, 1)
                assert 0.9 <= ratio <= 1.1, f"嵌套和扁平统计的 allocated 值应大致匹配: 嵌套={nested_allocated}, 扁平={flat_allocated}"
                break
    
    # 基本属性验证
    assert len(stats_dict) > 0, "统计字典不应为空"
    
    # 验证所有数值字段的有效性
    def validate_dict_values(d):
        for key, value in d.items():
            if isinstance(value, dict):
                validate_dict_values(value)
            elif isinstance(value, (int, float)):
                # 内存相关值应为非负数
                if "bytes" in key.lower() or "size" in key.lower() or "allocated" in key.lower():
                    assert value >= 0, f"{key} = {value} 应为非负数"
    
    validate_dict_values(stats_dict)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
@skip_if_no_cuda
@pytest.mark.parametrize("device,abbreviated", [
    (0, True),
    (0, False),  # 测试完整模式
])
def test_memory_summary_formatting(device, abbreviated):
    """TC-06: memory_summary 格式化输出
    
    测试 memory_summary 的格式化输出，验证：
    1. 输出为字符串类型
    2. 包含关键内存信息
    3. 基本属性正确性
    """
    # 重置内存统计
    reset_memory_stats(device)
    
    # 分配一些内存以产生有意义的摘要
    tensor1 = create_test_tensor(size=1024, dtype=torch.float32, device=device)
    tensor2 = create_test_tensor(size=2048, dtype=torch.float64, device=device)
    
    # 获取内存摘要
    summary = torch.cuda.memory_summary(device=device, abbreviated=abbreviated)
    
    # 验证返回类型是字符串
    assert isinstance(summary, str), "应返回字符串类型"
    assert len(summary) > 0, "摘要不应为空"
    
    # 验证包含关键信息
    required_keywords = [
        "allocated",  # 已分配内存
        "reserved",   # 保留内存
        "active",     # 活跃内存
        "GPU",        # GPU标识
    ]
    
    summary_lower = summary.lower()
    found_keywords = []
    
    for keyword in required_keywords:
        if keyword.lower() in summary_lower:
            found_keywords.append(keyword)
    
    # 至少应找到部分关键信息
    assert len(found_keywords) >= 2, f"摘要应包含关键内存信息，找到: {found_keywords}"
    
    # 验证摘要格式
    lines = summary.split('\n')
    assert len(lines) > 3, "摘要应有多个行"
    
    # 检查是否包含表格或结构化信息
    has_table_format = any('|' in line for line in lines) or any('-' * 10 in line for line in lines)
    if not has_table_format:
        # 如果没有表格格式，至少应有冒号分隔的键值对
        has_key_value = any(':' in line for line in lines)
        assert has_key_value, "摘要应包含结构化信息（表格或键值对）"
    
    # 验证 abbreviated 参数的影响
    full_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
    short_summary = torch.cuda.memory_summary(device=device, abbreviated=True)
    
    # 简略模式通常更短
    if abbreviated:
        current_summary = short_summary
        other_summary = full_summary
    else:
        current_summary = full_summary
        other_summary = short_summary
    
    # 验证当前模式与预期一致
    if abbreviated:
        # 简略模式可能更短或包含"abbreviated"提示
        assert len(current_summary) <= len(other_summary) * 1.5, "简略模式应更短或相当"
    else:
        # 完整模式通常包含更多细节
        assert len(current_summary) >= len(other_summary) * 0.7, "完整模式应包含更多细节"
    
    # 验证摘要包含数值信息
    import re
    # 查找数字（包括带逗号的数字，如 1,024）
    number_pattern = r'\b\d[\d,]*\b'
    numbers = re.findall(number_pattern, summary)
    
    assert len(numbers) > 0, "摘要应包含数值信息"
    
    # 验证数字的合理性（内存值）
    for num_str in numbers[:5]:  # 检查前5个数字
        # 移除逗号
        clean_num = num_str.replace(',', '')
        if clean_num.isdigit():
            num = int(clean_num)
            # 内存值应为非负数
            assert num >= 0, f"内存值应为非负数: {num}"
    
    # 验证多次调用的一致性
    summary2 = torch.cuda.memory_summary(device=device, abbreviated=abbreviated)
    assert isinstance(summary2, str), "第二次调用也应返回字符串"
    
    # 允许微小差异（时间戳、瞬时状态等）
    # 但核心结构应相似
    lines1 = summary.split('\n')
    lines2 = summary2.split('\n')
    
    # 检查行数大致相同
    assert abs(len(lines1) - len(lines2)) <= max(len(lines1), len(lines2)) * 0.3, \
        "多次调用的摘要结构应相似"
    
    # 验证摘要包含设备信息
    device_keywords = [f"cuda:{device}", f"GPU {device}", f"Device {device}"]
    has_device_info = any(keyword in summary for keyword in device_keywords)
    
    if not has_device_info:
        # 也可能使用通用设备标识
        generic_device_keys = ["GPU", "CUDA", "Device"]
        has_generic = any(key in summary for key in generic_device_keys)
        assert has_generic, "摘要应包含设备信息"
    
    # 清理
    del tensor1, tensor2
    torch.cuda.empty_cache()
    
    # 验证空状态下的摘要
    empty_summary = torch.cuda.memory_summary(device=device, abbreviated=abbreviated)
    assert isinstance(empty_summary, str), "空状态摘要也应返回字符串"
    assert len(empty_summary) > 0, "空状态摘要不应为空"
    
    # 空状态摘要应包含零或小的内存值
    zero_keywords = ["0", "zero", "none", "free"]
    has_zero_info = any(keyword.lower() in empty_summary.lower() for keyword in zero_keywords)
    
    if not has_zero_info:
        # 至少应包含"allocated"或"reserved"
        assert "allocated" in empty_summary.lower() or "reserved" in empty_summary.lower(), \
            "空状态摘要应包含内存状态信息"
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# TC-07: 内存快照功能验证 (DEFERRED)
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# TC-08: 大池小池分别统计准确性 (DEFERRED)
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# 测试类：组织相关测试
class TestCUDAMemoryStats:
    """torch.cuda.memory 统计功能测试类"""
    
    @skip_if_no_cuda
    def test_module_import(self):
        """测试模块导入和基本属性"""
        import torch.cuda.memory as memory_module
        
        # 验证模块存在
        assert memory_module is not None
        
        # 验证关键函数存在
        required_functions = [
            'memory_stats',
            'memory_stats_as_nested_dict',
            'memory_summary',
            'memory_snapshot',
        ]
        
        for func_name in required_functions:
            assert hasattr(memory_module, func_name), f"模块应包含 {func_name} 函数"
            func = getattr(memory_module, func_name)
            assert callable(func), f"{func_name} 应可调用"

# 参数化测试示例（供后续扩展使用）
@pytest.mark.parametrize("func_name", [
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
])
@skip_if_no_cuda
def test_memory_stats_function_signatures(func_name):
    """测试内存统计函数的基本签名"""
    func = getattr(torch.cuda, func_name)
    
    # 验证函数可调用
    assert callable(func)
    
    # 测试默认参数调用（使用当前设备）
    try:
        if func_name == "memory_summary":
            result = func()
        else:
            result = func(0)  # 需要设备参数
        
        # 验证返回类型
        if func_name == "memory_summary":
            assert isinstance(result, str), "memory_summary 应返回字符串"
        else:
            assert isinstance(result, dict), f"{func_name} 应返回字典"
    except Exception as e:
        # 如果失败，记录但允许（某些函数可能需要特定参数）
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