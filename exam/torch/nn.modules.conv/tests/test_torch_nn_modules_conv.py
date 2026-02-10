import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

# ==== BLOCK:HEADER START ====
"""
Test module for torch.nn.modules.conv
Generated based on test_plan.json
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


class TestConvModules:
    """Test class for torch.nn.modules.conv"""
    
    def setup_method(self):
        """Setup method for each test"""
        set_random_seed()
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
    @pytest.mark.parametrize(
        "conv_type,in_channels,out_channels,kernel_size,dtype,device,input_shape",
        [
            (
                "Conv2d",
                3,
                16,
                3,
                torch.float32,
                "cpu",
                (2, 3, 32, 32),
            ),
        ],
    )
    def test_conv2d_basic_instantiation_and_forward(
        self,
        conv_type,
        in_channels,
        out_channels,
        kernel_size,
        dtype,
        device,
        input_shape,
        random_seed,
    ):
        """
        TC-01: Conv2d 基本实例化与前向传播
        """
        # 1. 实例化模块
        if conv_type == "Conv2d":
            conv_module = nn.Conv2d(
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
        assert conv_module.kernel_size == (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size, \
            f"kernel_size should be {kernel_size}"
        
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
        expected_h = input_shape[2] - kernel_size + 1
        expected_w = input_shape[3] - kernel_size + 1
        expected_shape = (input_shape[0], out_channels, expected_h, expected_w)
        
        assert output.shape == expected_shape, \
            f"Output shape should be {expected_shape}, got {output.shape}"
        
        # 7. 验证值有限性
        assert torch.isfinite(output).all(), "Output should contain only finite values"
        
        # 8. 使用torch.nn.functional.conv2d作为oracle验证
        with torch.no_grad():
            # 获取卷积权重和偏置
            weight = conv_module.weight
            bias = conv_module.bias
            
            # 使用functional.conv2d计算
            expected_output = F.conv2d(
                x,
                weight,
                bias,
                stride=conv_module.stride,
                padding=conv_module.padding,
                dilation=conv_module.dilation,
                groups=conv_module.groups,
            )
            
            # 比较结果（使用容差）
            torch.testing.assert_close(
                output,
                expected_output,
                rtol=1e-5,
                atol=1e-5,
                msg="Output should match functional.conv2d result",
            )
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
    @pytest.mark.parametrize(
        "test_type,in_channels,out_channels,kernel_size,groups,padding_mode,expect_error",
        [
            (
                "invalid_groups",
                8,
                16,
                3,
                3,  # groups=3 不整除 in_channels=8
                "zeros",
                ValueError,
            ),
            (
                "invalid_padding_mode",
                3,
                16,
                3,
                1,
                "invalid",  # 无效的padding_mode
                ValueError,
            ),
        ],
    )
    def test_parameter_validation_and_exception_handling(
        self,
        test_type,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        padding_mode,
        expect_error,
        random_seed,
    ):
        """
        TC-02: 参数验证与异常处理
        """
        # 验证异常被正确抛出
        with pytest.raises(expect_error) as exc_info:
            if test_type == "invalid_groups":
                # groups 不整除 in_channels 的情况
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    groups=groups,
                )
            elif test_type == "invalid_padding_mode":
                # 无效 padding_mode 的情况
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                )
            else:
                raise ValueError(f"Unknown test_type: {test_type}")
        
        # 验证错误消息包含关键信息
        error_msg = str(exc_info.value).lower()
        
        if test_type == "invalid_groups":
            # 检查错误消息是否提到 groups 或整除性
            assert "groups" in error_msg or "divisible" in error_msg, \
                f"Error message should mention groups or divisibility, got: {error_msg}"
            
            # 验证具体的错误条件
            # groups 应该整除 in_channels 和 out_channels
            if groups > in_channels:
                assert "greater" in error_msg or "exceed" in error_msg, \
                    f"Error message should mention groups exceeding in_channels, got: {error_msg}"
        
        elif test_type == "invalid_padding_mode":
            # 检查错误消息是否提到 padding_mode
            assert "padding_mode" in error_msg or "padding mode" in error_msg, \
                f"Error message should mention padding_mode, got: {error_msg}"
            
            # 检查是否提到了允许的值
            allowed_modes = ["zeros", "reflect", "replicate", "circular"]
            for mode in allowed_modes:
                if mode in error_msg:
                    break
            else:
                # 如果没有提到任何允许的模式，至少应该提到"invalid"或"not supported"
                assert "invalid" in error_msg or "not supported" in error_msg or "not one of" in error_msg, \
                    f"Error message should mention invalid or unsupported, got: {error_msg}"
        
        # 验证异常类型正确
        assert isinstance(exc_info.value, expect_error), \
            f"Exception type should be {expect_error}, got {type(exc_info.value)}"
        
        # 验证没有其他异常被意外抛出
        # （通过with pytest.raises已经确保）
# ==== BLOCK:CASE_02 END ====

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
        
        # 检查 kernel_size 是否正确设置
        if isinstance(kernel_size, int):
            if conv_type == "Conv1d":
                assert conv_module.kernel_size == (kernel_size,), f"kernel_size should be ({kernel_size},)"
            elif conv_type == "Conv3d":
                assert conv_module.kernel_size == (kernel_size, kernel_size, kernel_size), \
                    f"kernel_size should be ({kernel_size}, {kernel_size}, {kernel_size})"
        
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
            
            # 使用 functional.conv1d 作为 oracle 验证
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
            
            # 使用 functional.conv3d 作为 oracle 验证
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
        
        # 8. 验证梯度流（弱断言）
        # 创建损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 验证权重梯度存在
        assert conv_module.weight.grad is not None, "Weight gradients should be computed"
        assert torch.isfinite(conv_module.weight.grad).all(), "Weight gradients should be finite"
        
        # 如果有偏置，验证偏置梯度
        if conv_module.bias is not None:
            assert conv_module.bias.grad is not None, "Bias gradients should be computed"
            assert torch.isfinite(conv_module.bias.grad).all(), "Bias gradients should be finite"
        
        # 9. 验证跨维度的一致性
        # 对于 Conv1d 和 Conv3d，验证它们的基本行为与 Conv2d 一致
        # （通过成功实例化、前向传播和梯度计算已经验证）
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
    @pytest.mark.parametrize(
        "padding,stride,kernel_size,input_shape,padding_mode",
        [
            (
                "same",
                1,
                3,
                (2, 3, 32, 32),
                "zeros",
            ),
            (
                1,
                1,
                3,
                (2, 3, 32, 32),
                "reflect",
            ),
        ],
    )
    def test_padding_same_and_special_modes(
        self,
        padding,
        stride,
        kernel_size,
        input_shape,
        padding_mode,
        random_seed,
    ):
        """
        TC-04: padding='same' 与特殊模式
        """
        # 1. 实例化模块
        conv_module = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=16,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )
        
        # 2. 验证模块属性
        assert conv_module is not None, "Module should be instantiated"
        assert conv_module.padding_mode == padding_mode, \
            f"padding_mode should be {padding_mode}, got {conv_module.padding_mode}"
        
        # 3. 准备输入数据
        x = torch.randn(*input_shape, dtype=torch.float32)
        
        # 4. 前向传播
        output = conv_module(x)
        
        # 5. 验证输出
        assert output is not None, "Forward pass should produce output"
        assert torch.isfinite(output).all(), "Output should contain only finite values"
        
        # 6. 验证输出形状
        if padding == "same":
            # padding='same' 时，输出形状应该与输入形状相同（除了通道数）
            # 公式: out_size = ceil(in_size / stride)
            # 对于 stride=1，输出大小应该等于输入大小
            expected_h = input_shape[2]
            expected_w = input_shape[3]
            expected_shape = (input_shape[0], 16, expected_h, expected_w)
            
            assert output.shape == expected_shape, \
                f"With padding='same' and stride={stride}, output shape should be {expected_shape}, got {output.shape}"
            
            # 验证 stride=1 的条件（padding='same' 要求 stride=1）
            assert stride == 1, "padding='same' requires stride=1"
        
        else:
            # 普通 padding 情况
            # 公式: out_size = floor((in_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
            # 默认 dilation=1
            padding_val = padding if isinstance(padding, int) else 0
            expected_h = math.floor((input_shape[2] + 2 * padding_val - (kernel_size - 1) - 1) / stride + 1)
            expected_w = math.floor((input_shape[3] + 2 * padding_val - (kernel_size - 1) - 1) / stride + 1)
            expected_shape = (input_shape[0], 16, expected_h, expected_w)
            
            assert output.shape == expected_shape, \
                f"Output shape should be {expected_shape}, got {output.shape}"
        
        # 7. 特殊验证：reflect padding 模式
        if padding_mode == "reflect":
            # reflect padding 应该能正常工作
            # 验证输出值有限且没有 NaN/Inf
            assert not torch.isnan(output).any(), "Output should not contain NaN values"
            assert not torch.isinf(output).any(), "Output should not contain Inf values"
            
            # 验证输出值在合理范围内
            # reflect padding 可能导致值稍微变大，但应该在合理范围内
            output_abs_max = torch.abs(output).max().item()
            assert output_abs_max < 100.0, \
                f"Output values seem too large for reflect padding: max abs value = {output_abs_max}"
        
        # 8. 验证没有异常抛出
        # （通过成功执行到此处已经验证）
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
    @pytest.mark.parametrize(
        "in_channels,out_channels,kernel_size,bias,test_samples",
        [
            (
                3,
                16,
                3,
                True,
                1000,
            ),
        ],
    )
    def test_weight_initialization_verification(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias,
        test_samples,
        random_seed,
    ):
        """
        TC-05: 权重初始化验证
        """
        # 1. 创建多个 Conv2d 实例来验证权重初始化分布
        modules = []
        weight_samples = []
        bias_samples = []
        
        for i in range(test_samples):
            # 每次创建新模块时重置随机种子以确保独立性
            torch.manual_seed(42 + i)
            
            conv_module = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            )
            modules.append(conv_module)
            
            # 收集权重样本
            weight_samples.append(conv_module.weight.detach().flatten())
            
            # 收集偏置样本（如果启用）
            if bias:
                bias_samples.append(conv_module.bias.detach().flatten())
        
        # 2. 验证权重已初始化
        for i, module in enumerate(modules):
            assert module.weight is not None, f"Weight should be initialized for module {i}"
            assert torch.isfinite(module.weight).all(), f"Weight should contain only finite values for module {i}"
            
            if bias:
                assert module.bias is not None, f"Bias should be initialized for module {i}"
                assert torch.isfinite(module.bias).all(), f"Bias should contain only finite values for module {i}"
        
        # 3. 合并所有样本
        all_weights = torch.cat(weight_samples)
        
        # 4. 验证均匀分布（弱断言）
        # 计算理论范围：U(-√k, √k)，其中 k = 1/(in_channels * kernel_size * kernel_size)
        k = 1.0 / (in_channels * kernel_size * kernel_size)
        bound = math.sqrt(k)
        
        # 验证所有权重都在理论范围内
        assert torch.all(all_weights >= -bound), f"All weights should be >= {-bound}, min={all_weights.min().item()}"
        assert torch.all(all_weights <= bound), f"All weights should be <= {bound}, max={all_weights.max().item()}"
        
        # 5. 验证均值和方差（弱断言）
        mean_val = all_weights.mean().item()
        std_val = all_weights.std().item()
        
        # 均匀分布 U(-a, a) 的理论均值为0，方差为 a²/3
        expected_variance = bound * bound / 3.0
        expected_std = math.sqrt(expected_variance)
        
        # 验证均值接近0（容差较大）
        assert abs(mean_val) < 0.1, f"Weight mean should be close to 0, got {mean_val}"
        
        # 验证标准差在合理范围内
        assert 0.5 * expected_std < std_val < 1.5 * expected_std, \
            f"Weight std should be around {expected_std}, got {std_val}"
        
        # 6. 验证偏置初始化（如果启用）
        if bias:
            all_biases = torch.cat(bias_samples)
            
            # 偏置通常初始化为0或小常数
            # 验证偏置值在合理范围内
            bias_mean = all_biases.mean().item()
            bias_std = all_biases.std().item()
            
            # 偏置通常初始化为0或接近0
            assert abs(bias_mean) < 0.01, f"Bias mean should be close to 0, got {bias_mean}"
            assert bias_std < 0.01, f"Bias std should be small, got {bias_std}"
            
            # 验证偏置值有限
            assert torch.isfinite(all_biases).all(), "All bias values should be finite"
        
        # 7. 验证不同模块的权重不同（由于随机初始化）
        if test_samples > 1:
            # 比较前两个模块的权重
            weight_diff = torch.abs(modules[0].weight - modules[1].weight).sum().item()
            assert weight_diff > 1e-6, "Different modules should have different weights due to random initialization"
        
        # 8. 验证权重形状正确
        expected_weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        for i, module in enumerate(modules):
            assert module.weight.shape == expected_weight_shape, \
                f"Weight shape should be {expected_weight_shape}, got {module.weight.shape} for module {i}"
            
            if bias:
                expected_bias_shape = (out_channels,)
                assert module.bias.shape == expected_bias_shape, \
                    f"Bias shape should be {expected_bias_shape}, got {module.bias.shape} for module {i}"
        
        # 9. 验证没有 NaN 或 Inf 值
        assert not torch.isnan(all_weights).any(), "Weights should not contain NaN values"
        assert not torch.isinf(all_weights).any(), "Weights should not contain Inf values"
        
        if bias:
            assert not torch.isnan(all_biases).any(), "Biases should not contain NaN values"
            assert not torch.isinf(all_biases).any(), "Biases should not contain Inf values"
        
        # 10. 验证权重值在合理范围内（额外检查）
        weight_abs_max = torch.abs(all_weights).max().item()
        assert weight_abs_max <= bound * 1.01, \
            f"Weight absolute max should be <= {bound * 1.01}, got {weight_abs_max}"
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
    # TC-06: 占位用例 (DEFERRED_SET，首轮仅占位)
    # 将在后续迭代中实现
    pass
# ==== BLOCK:CASE_06 END ====

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