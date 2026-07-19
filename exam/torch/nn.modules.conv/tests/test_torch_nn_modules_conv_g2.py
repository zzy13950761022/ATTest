import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

# ==== BLOCK:HEADER START ====
"""
Test module for torch.nn.modules.conv - G2 Group
Generated based on test_plan.json
Group G2: 卷积变体与高级功能
Entrypoints: Conv1d, Conv3d, ConvTranspose2d
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock


def set_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(scope="function")
def random_seed():
    """Fixture to set random seed for each test"""
    set_random_seed()
    return 42


class TestConvModulesG2:
    """Test class for torch.nn.modules.conv - G2 Group"""
    
    def setup_method(self):
        """Setup method for each test"""
        set_random_seed()
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
    @pytest.mark.parametrize(
        "conv_type,in_channels,out_channels,kernel_size,input_shape,dtype,device,stride,padding,dilation,groups,bias",
        [
            # 基本测试用例
            (
                "Conv1d",
                3,
                16,
                3,
                (2, 3, 32),
                torch.float32,
                "cpu",
                1,
                0,
                1,
                1,
                True,
            ),
            (
                "Conv3d",
                3,
                16,
                3,
                (2, 3, 16, 16, 16),
                torch.float32,
                "cpu",
                1,
                0,
                1,
                1,
                True,
            ),
            (
                "ConvTranspose2d",
                3,
                16,
                3,
                (2, 3, 16, 16),
                torch.float32,
                "cpu",
                1,
                0,
                1,
                1,
                True,
            ),
            # Conv3d边界条件测试 - 修复行133-135覆盖率缺口
            (
                "Conv3d",
                1,
                1,
                1,
                (1, 1, 4, 4, 4),
                torch.float32,
                "cpu",
                1,
                0,
                1,
                1,
                False,
            ),
            # 非默认参数测试 - 修复分支259->262覆盖率缺口
            (
                "Conv1d",
                4,
                8,
                5,
                (2, 4, 64),
                torch.float32,
                "cpu",
                2,
                2,
                2,
                2,
                False,
            ),
            # CUDA设备测试（如果可用）- 修复分支214->247覆盖率缺口
            (
                "Conv3d",
                2,
                4,
                2,
                (1, 2, 8, 8, 8),
                torch.float32,
                "cuda" if torch.cuda.is_available() else "cpu",
                1,
                1,
                1,
                1,
                True,
            ),
        ],
    )
    def test_conv1d_and_conv3d_basic_functionality(
        self,
        conv_type,
        in_channels,
        out_channels,
        kernel_size,
        input_shape,
        dtype,
        device,
        stride,
        padding,
        dilation,
        groups,
        bias,
        random_seed,
    ):
        """
        TC-03: Conv1d 和 Conv3d 基本功能
        包含扩展：ConvTranspose2d 测试
        修复覆盖率缺口：
        1. Conv3d边界条件测试（行133-135）
        2. CUDA设备条件测试（分支214->247）
        3. 非默认参数条件测试（分支259->262）
        """
        # 跳过CUDA测试如果设备不可用
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # 1. 实例化模块
        conv_args = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "dtype": dtype,
        }
        
        if conv_type == "Conv1d":
            conv_module = nn.Conv1d(**conv_args)
        elif conv_type == "Conv3d":
            conv_module = nn.Conv3d(**conv_args)
        elif conv_type == "ConvTranspose2d":
            conv_module = nn.ConvTranspose2d(**conv_args)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
        
        # 2. 验证模块属性
        assert conv_module is not None, "Module should be instantiated"
        assert conv_module.in_channels == in_channels, f"in_channels should be {in_channels}"
        assert conv_module.out_channels == out_channels, f"out_channels should be {out_channels}"
        
        # 修复：正确处理不同卷积类型的stride属性
        # Conv1d的stride是元组(1,)，Conv2d/ConvTranspose2d是(1,1)，Conv3d是(1,1,1)
        if conv_type == "Conv1d":
            expected_stride = (stride,)
        elif conv_type == "ConvTranspose2d":
            expected_stride = (stride, stride)
        elif conv_type == "Conv3d":
            expected_stride = (stride, stride, stride)
        else:
            expected_stride = (stride, stride)  # 默认Conv2d
        
        assert conv_module.stride == expected_stride, \
            f"stride should be {expected_stride}, got {conv_module.stride}"
        
        # 修复：正确处理不同卷积类型的padding属性
        if conv_type == "Conv1d":
            expected_padding = (padding,)
        elif conv_type == "ConvTranspose2d":
            expected_padding = (padding, padding)
        elif conv_type == "Conv3d":
            expected_padding = (padding, padding, padding)
        else:
            expected_padding = (padding, padding)  # 默认Conv2d
        
        assert conv_module.padding == expected_padding, \
            f"padding should be {expected_padding}, got {conv_module.padding}"
        
        # 修复：正确处理不同卷积类型的dilation属性
        if conv_type == "Conv1d":
            expected_dilation = (dilation,)
        elif conv_type == "ConvTranspose2d":
            expected_dilation = (dilation, dilation)
        elif conv_type == "Conv3d":
            expected_dilation = (dilation, dilation, dilation)
        else:
            expected_dilation = (dilation, dilation)  # 默认Conv2d
        
        assert conv_module.dilation == expected_dilation, \
            f"dilation should be {expected_dilation}, got {conv_module.dilation}"
        
        assert conv_module.groups == groups, f"groups should be {groups}"
        assert conv_module.bias is not None if bias else conv_module.bias is None, \
            f"bias should be {'enabled' if bias else 'disabled'}"
        
        # 检查kernel_size是否正确设置
        if isinstance(kernel_size, int):
            if conv_type == "Conv1d":
                expected_kernel_size = (kernel_size,)
            elif conv_type in ["Conv2d", "ConvTranspose2d"]:
                expected_kernel_size = (kernel_size, kernel_size)
            elif conv_type == "Conv3d":
                expected_kernel_size = (kernel_size, kernel_size, kernel_size)
            else:
                expected_kernel_size = (kernel_size,) * getattr(conv_module, '_get_dim', lambda: 2)()
        else:
            expected_kernel_size = kernel_size
        
        assert conv_module.kernel_size == expected_kernel_size, \
            f"kernel_size should be {expected_kernel_size}, got {conv_module.kernel_size}"
        
        # 3. 准备输入数据
        x = torch.randn(*input_shape, dtype=dtype)
        if device == "cuda":
            x = x.cuda()
        
        # 4. 前向传播
        output = conv_module(x)
        
        # 5. 验证输出
        assert output is not None, "Forward pass should produce output"
        assert output.dtype == dtype, f"Output dtype should be {dtype}"
        if device == "cuda":
            assert output.is_cuda, "Output should be on CUDA device"
        
        # 6. 验证输出形状
        # 计算期望的输出形状
        # 对于普通卷积：out_size = floor((in_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
        # 对于转置卷积：out_size = (in_size - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
        # 默认参数: padding=0, stride=1, dilation=1, output_padding=0
        
        if conv_type == "Conv1d":
            # Conv1d: 输入形状为 (batch, channels, length)
            expected_length = math.floor((input_shape[2] + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
            expected_shape = (input_shape[0], out_channels, expected_length)
            
            # 使用functional.conv1d作为oracle验证
            with torch.no_grad():
                weight = conv_module.weight
                bias_tensor = conv_module.bias
                
                expected_output = F.conv1d(
                    x,
                    weight,
                    bias_tensor,
                    stride=conv_module.stride,
                    padding=conv_module.padding,
                    dilation=conv_module.dilation,
                    groups=conv_module.groups,
                )
                
                torch.testing.assert_close(
                    output,
                    expected_output,
                    rtol=1e-5,
                    atol=1e-5,
                    msg="Conv1d output should match functional.conv1d result",
                )
                
        elif conv_type == "Conv3d":
            # Conv3d: 输入形状为 (batch, channels, depth, height, width)
            expected_depth = math.floor((input_shape[2] + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
            expected_height = math.floor((input_shape[3] + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
            expected_width = math.floor((input_shape[4] + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
            expected_shape = (input_shape[0], out_channels, expected_depth, expected_height, expected_width)
            
            # 使用functional.conv3d作为oracle验证
            with torch.no_grad():
                weight = conv_module.weight
                bias_tensor = conv_module.bias
                
                expected_output = F.conv3d(
                    x,
                    weight,
                    bias_tensor,
                    stride=conv_module.stride,
                    padding=conv_module.padding,
                    dilation=conv_module.dilation,
                    groups=conv_module.groups,
                )
                
                torch.testing.assert_close(
                    output,
                    expected_output,
                    rtol=1e-5,
                    atol=1e-5,
                    msg="Conv3d output should match functional.conv3d result",
                )
        
        elif conv_type == "ConvTranspose2d":
            # ConvTranspose2d: 输入形状为 (batch, channels, height, width)
            # 转置卷积的输出尺寸公式：
            # out_size = (in_size - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
            # 默认参数: stride=1, padding=0, dilation=1, output_padding=0
            expected_height = (input_shape[2] - 1) * stride - 2*padding + dilation*(kernel_size-1) + 1
            expected_width = (input_shape[3] - 1) * stride - 2*padding + dilation*(kernel_size-1) + 1
            expected_shape = (input_shape[0], out_channels, expected_height, expected_width)
            
            # 使用functional.conv_transpose2d作为oracle验证
            with torch.no_grad():
                weight = conv_module.weight
                bias_tensor = conv_module.bias
                
                expected_output = F.conv_transpose2d(
                    x,
                    weight,
                    bias_tensor,
                    stride=conv_module.stride,
                    padding=conv_module.padding,
                    output_padding=conv_module.output_padding,
                    groups=conv_module.groups,
                    dilation=conv_module.dilation,
                )
                
                torch.testing.assert_close(
                    output,
                    expected_output,
                    rtol=1e-5,
                    atol=1e-5,
                    msg="ConvTranspose2d output should match functional.conv_transpose2d result",
                )
        
        assert output.shape == expected_shape, \
            f"Output shape should be {expected_shape}, got {output.shape}"
        
        # 7. 验证值有限性
        assert torch.isfinite(output).all(), "Output should contain only finite values"
        
        # 8. 验证不同维度的一致性
        # 对于Conv1d、Conv3d和ConvTranspose2d，验证它们的基本行为一致
        # 主要验证模块能正确实例化和前向传播
        
        # 9. 验证梯度流（弱断言级别）
        # 检查模块参数是否有梯度
        if conv_module.bias is not None:
            assert conv_module.bias.requires_grad, "Bias should require gradient"
        
        assert conv_module.weight.requires_grad, "Weight should require gradient"
        
        # 10. 验证输出值在合理范围内
        output_mean = output.mean().item()
        output_std = output.std().item()
        
        # 输出值应该在合理范围内（基于随机输入）
        # 对于转置卷积，输出值可能稍大，所以放宽限制
        if conv_type == "ConvTranspose2d":
            assert abs(output_mean) < 10.0, f"ConvTranspose2d output mean seems too large: {output_mean}"
            assert 0.1 < output_std < 20.0, f"ConvTranspose2d output std seems unreasonable: {output_std}"
        else:
            assert abs(output_mean) < 5.0, f"Output mean seems too large: {output_mean}"
            assert 0.1 < output_std < 10.0, f"Output std seems unreasonable: {output_std}"
        
        # 11. 验证转置卷积的特殊属性
        if conv_type == "ConvTranspose2d":
            # 验证output_padding属性
            assert hasattr(conv_module, 'output_padding'), "ConvTranspose2d should have output_padding attribute"
            assert conv_module.output_padding == (0, 0) or conv_module.output_padding == 0, \
                f"Default output_padding should be 0, got {conv_module.output_padding}"
        
        # 12. 验证边界条件
        if conv_type == "Conv3d" and kernel_size == 1 and in_channels == 1 and out_channels == 1:
            # 对于1x1x1卷积，输出应该与输入形状相同（除了通道数）
            assert output.shape[2:] == input_shape[2:], \
                f"For 1x1x1 convolution, spatial dimensions should match input"
        
        # 13. 验证非默认参数
        if stride != 1 or padding != 0 or dilation != 1 or groups != 1:
            # 验证非默认参数确实被使用
            if stride != 1:
                # 检查stride是否不是默认值
                default_stride = (1,) if conv_type == "Conv1d" else (1, 1) if conv_type == "ConvTranspose2d" else (1, 1, 1)
                assert conv_module.stride != default_stride, \
                    "Non-default stride should be set"
            
            if padding != 0:
                # 检查padding是否不是默认值
                default_padding = (0,) if conv_type == "Conv1d" else (0, 0) if conv_type == "ConvTranspose2d" else (0, 0, 0)
                assert conv_module.padding != default_padding, \
                    "Non-default padding should be set"
            
            if dilation != 1:
                # 检查dilation是否不是默认值
                default_dilation = (1,) if conv_type == "Conv1d" else (1, 1) if conv_type == "ConvTranspose2d" else (1, 1, 1)
                assert conv_module.dilation != default_dilation, \
                    "Non-default dilation should be set"
            
            if groups != 1:
                assert conv_module.groups != 1, "Non-default groups should be set"
        
        # 14. 验证CUDA设备（如果使用）
        if device == "cuda" and torch.cuda.is_available():
            assert conv_module.weight.is_cuda, "Module weights should be on CUDA"
            if conv_module.bias is not None:
                assert conv_module.bias.is_cuda, "Module bias should be on CUDA"
            assert x.is_cuda, "Input should be on CUDA"
            assert output.is_cuda, "Output should be on CUDA"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_07 START ====
    # TC-07: 占位用例 (DEFERRED_SET，首轮仅占位)
    # 将在后续迭代中实现
    pass
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
    # TC-08: 占位用例 (DEFERRED_SET，首轮仅占位)
    # 将在后续迭代中实现
    pass
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:CASE_09 START ====
    # TC-09: 占位用例 (DEFERRED_SET，首轮仅占位)
    # 将在后续迭代中实现
    pass
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:FOOTER START ====
# ==== BLOCK:FOOTER END ====