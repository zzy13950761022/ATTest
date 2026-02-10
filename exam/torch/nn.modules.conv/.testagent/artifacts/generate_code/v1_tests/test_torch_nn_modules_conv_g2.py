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
        "conv_type,in_channels,out_channels,kernel_size,input_shape,dtype,device",
        [
            (
                "Conv1d",
                3,
                16,
                3,
                (2, 3, 32),
                torch.float32,
                "cpu",
            ),
            (
                "Conv3d",
                3,
                16,
                3,
                (2, 3, 16, 16, 16),
                torch.float32,
                "cpu",
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
        random_seed,
    ):
        """
        TC-03: Conv1d 和 Conv3d 基本功能
        """
        # 1. 实例化模块
        if conv_type == "Conv1d":
            conv_module = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dtype=dtype,
            )
        elif conv_type == "Conv3d":
            conv_module = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
        
        # 2. 验证模块属性
        assert conv_module is not None, "Module should be instantiated"
        assert conv_module.in_channels == in_channels, f"in_channels should be {in_channels}"
        assert conv_module.out_channels == out_channels, f"out_channels should be {out_channels}"
        
        # 检查kernel_size是否正确设置
        if isinstance(kernel_size, int):
            expected_kernel_size = (kernel_size,) * conv_module._get_dim()
        else:
            expected_kernel_size = kernel_size
        
        assert conv_module.kernel_size == expected_kernel_size, \
            f"kernel_size should be {expected_kernel_size}, got {conv_module.kernel_size}"
        
        # 3. 准备输入数据
        x = torch.randn(*input_shape, dtype=dtype, device=device)
        
        # 4. 前向传播
        output = conv_module(x)
        
        # 5. 验证输出
        assert output is not None, "Forward pass should produce output"
        assert output.dtype == dtype, f"Output dtype should be {dtype}"
        
        # 6. 验证输出形状
        # 计算期望的输出形状
        # 公式: out_size = floor((in_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
        # 默认参数: padding=0, stride=1, dilation=1
        
        if conv_type == "Conv1d":
            # Conv1d: 输入形状为 (batch, channels, length)
            expected_length = input_shape[2] - kernel_size + 1
            expected_shape = (input_shape[0], out_channels, expected_length)
            
            # 使用functional.conv1d作为oracle验证
            with torch.no_grad():
                weight = conv_module.weight
                bias = conv_module.bias
                
                expected_output = F.conv1d(
                    x,
                    weight,
                    bias,
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
            expected_depth = input_shape[2] - kernel_size + 1
            expected_height = input_shape[3] - kernel_size + 1
            expected_width = input_shape[4] - kernel_size + 1
            expected_shape = (input_shape[0], out_channels, expected_depth, expected_height, expected_width)
            
            # 使用functional.conv3d作为oracle验证
            with torch.no_grad():
                weight = conv_module.weight
                bias = conv_module.bias
                
                expected_output = F.conv3d(
                    x,
                    weight,
                    bias,
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
        
        assert output.shape == expected_shape, \
            f"Output shape should be {expected_shape}, got {output.shape}"
        
        # 7. 验证值有限性
        assert torch.isfinite(output).all(), "Output should contain only finite values"
        
        # 8. 验证不同维度的一致性
        # 对于Conv1d和Conv3d，验证它们的基本行为与Conv2d一致
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
        assert abs(output_mean) < 5.0, f"Output mean seems too large: {output_mean}"
        assert 0.1 < output_std < 10.0, f"Output std seems unreasonable: {output_std}"
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