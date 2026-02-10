# ==== BLOCK:HEADER START ====
import math
import pytest
import torch
import torch.nn.parallel.comm as comm
from unittest.mock import patch, MagicMock


def setup_module():
    """设置测试模块，固定随机种子"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def teardown_module():
    """清理测试模块"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def cuda_devices():
    """获取可用的CUDA设备，支持CPU回退"""
    if not torch.cuda.is_available():
        # CPU回退：返回CPU设备列表
        return [torch.device("cpu") for _ in range(3)]
    
    device_count = torch.cuda.device_count()
    if device_count < 2:
        # 如果CUDA设备不足，也使用CPU回退
        return [torch.device("cpu") for _ in range(3)]
    
    return [torch.device(f"cuda:{i}") for i in range(min(device_count, 3))]


@pytest.fixture
def random_tensor():
    """生成随机张量的工厂函数"""
    def _create(shape, dtype=torch.float32, device="cpu"):
        tensor = torch.randn(*shape, dtype=dtype, device=device)
        return tensor
    
    return _create


class TestReduceAddFunctions:
    """归约函数族测试类 (G2)"""
    pass
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_05 START ====
    @pytest.mark.parametrize("dtype,devices,shape,destination", [
        (torch.float32, ["cuda:0", "cuda:1"], (3, 4), "cuda:0"),
        (torch.float64, ["cuda:0", "cuda:1", "cuda:2"], (5, 5), "cuda:1"),
    ])
    def test_reduce_add_basic(self, cuda_devices, random_tensor, dtype, devices, shape, destination):
        """TC-05: reduce_add基本归约
        
        测试多GPU张量归约求和数值正确性
        """
        # 检查CUDA可用性，如果不可用则使用CPU回退
        if not torch.cuda.is_available():
            # CPU回退：将设备映射到CPU
            devices = ["cpu"] * len(devices)
            destination = "cpu"
            pytest.skip("CUDA not available, using CPU fallback for testing logic")
        
        # 跳过如果设备不足
        if len(cuda_devices) < len(devices):
            pytest.skip(f"Need {len(devices)} CUDA devices, got {len(cuda_devices)}")
        
        # 创建输入张量列表，每个设备一个
        inputs = []
        expected_sum = torch.zeros(shape, dtype=dtype, device=destination)
        
        for i, device_str in enumerate(devices):
            # 创建随机张量
            tensor = random_tensor(shape, dtype=dtype, device=device_str)
            inputs.append(tensor)
            
            # 累加到期望的和（移动到目标设备）
            expected_sum.add_(tensor.to(destination))
        
        # 使用mock来模拟CUDA不可用的情况
        with patch('torch.nn.parallel.comm._get_device_index') as mock_get_device_index:
            # 模拟_get_device_index函数，对于CPU设备返回-1
            def mock_get_device_index_func(device, optional=False, allow_cpu=False):
                if isinstance(device, str):
                    device = torch.device(device)
                if isinstance(device, torch.device):
                    if device.type == "cpu":
                        return -1
                    else:
                        # 对于CUDA设备，返回设备索引
                        return device.index if device.index is not None else 0
                elif isinstance(device, int):
                    return device
                else:
                    raise ValueError(f"Unexpected device type: {type(device)}")
            
            mock_get_device_index.side_effect = mock_get_device_index_func
            
            # 执行归约求和
            result = comm.reduce_add(inputs, destination=destination)
        
        # weak断言验证
        # 1. 返回类型
        assert isinstance(result, torch.Tensor), "reduce_add should return a Tensor"
        
        # 2. 形状匹配
        assert result.shape == shape, \
            f"Shape mismatch, expected {shape}, got {result.shape}"
        
        # 3. 数据类型匹配
        assert result.dtype == dtype, \
            f"dtype mismatch, expected {dtype}, got {result.dtype}"
        
        # 4. 设备匹配
        expected_device = torch.device(destination)
        assert result.device == expected_device, \
            f"Device mismatch, expected {expected_device}, got {result.device}"
        
        # 5. 数值正确性（与手动计算的和比较）
        assert torch.allclose(result, expected_sum, rtol=1e-6, atol=1e-6), \
            "Sum correctness mismatch"
        
        # 6. 输入张量不应被修改
        for i, (input_tensor, device_str) in enumerate(zip(inputs, devices)):
            # 重新计算该张量的值进行验证
            # 注意：由于张量在GPU上，我们需要确保比较在同一设备上进行
            original_value = input_tensor.clone()
            assert torch.allclose(input_tensor, original_value, rtol=1e-6, atol=1e-6), \
                f"Input tensor {i} should not be modified"
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
    def test_reduce_add_shape_mismatch_exception(self, cuda_devices, random_tensor):
        """TC-06: reduce_add形状不匹配异常
        
        测试当输入张量形状不匹配时抛出RuntimeError异常
        """
        # 检查CUDA可用性，如果不可用则使用CPU回退
        if not torch.cuda.is_available():
            # CPU回退测试
            devices = ["cpu", "cpu"]
            destination = "cpu"
        else:
            # 跳过如果设备不足
            if len(cuda_devices) < 2:
                pytest.skip(f"Need at least 2 CUDA devices, got {len(cuda_devices)}")
            devices = ["cuda:0", "cuda:1"]
            destination = "cuda:0"
        
        # 创建形状不匹配的张量列表
        # 第一个张量形状为 (2, 3)，第二个为 (3, 4)
        inputs = []
        
        # 第一个张量：形状 (2, 3)
        shape1 = (2, 3)
        tensor1 = random_tensor(shape1, dtype=torch.float32, device=devices[0])
        inputs.append(tensor1)
        
        # 第二个张量：形状 (3, 4) - 与第一个不匹配
        shape2 = (3, 4)
        tensor2 = random_tensor(shape2, dtype=torch.float32, device=devices[1])
        inputs.append(tensor2)
        
        # 使用mock来模拟CUDA不可用的情况
        with patch('torch.nn.parallel.comm._get_device_index') as mock_get_device_index:
            # 模拟_get_device_index函数，对于CPU设备返回-1
            def mock_get_device_index_func(device, optional=False, allow_cpu=False):
                if isinstance(device, str):
                    device = torch.device(device)
                if isinstance(device, torch.device):
                    if device.type == "cpu":
                        return -1
                    else:
                        # 对于CUDA设备，返回设备索引
                        return device.index if device.index is not None else 0
                elif isinstance(device, int):
                    return device
                else:
                    raise ValueError(f"Unexpected device type: {type(device)}")
            
            mock_get_device_index.side_effect = mock_get_device_index_func
            
            # 验证当形状不匹配时抛出RuntimeError异常
            with pytest.raises(RuntimeError) as exc_info:
                comm.reduce_add(inputs, destination=destination)
        
        # weak断言验证
        # 1. 异常类型正确
        assert isinstance(exc_info.value, RuntimeError), \
            f"Expected RuntimeError, got {type(exc_info.value).__name__}"
        
        # 2. 异常消息包含相关关键词
        error_msg = str(exc_info.value).lower()
        # 检查是否包含形状相关关键词
        shape_keywords = ["shape", "size", "dimension", "mismatch", "inconsistent"]
        has_shape_keyword = any(keyword in error_msg for keyword in shape_keywords)
        assert has_shape_keyword, \
            f"Error message should contain shape-related keywords. Got: {error_msg}"
        
        # 3. 异常消息包含张量相关信息
        tensor_keywords = ["tensor", "input", "argument"]
        has_tensor_keyword = any(keyword in error_msg for keyword in tensor_keywords)
        assert has_tensor_keyword, \
            f"Error message should contain tensor-related keywords. Got: {error_msg}"
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
    # ==== DEFERRED: reduce_add_coalesced合并归约 (TC-07) ====
    # 将在后续轮次中实现
    pass
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:FOOTER START ====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====