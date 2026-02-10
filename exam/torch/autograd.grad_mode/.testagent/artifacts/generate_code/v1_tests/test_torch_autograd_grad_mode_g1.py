# G1组测试文件：基础梯度控制类
# 包含测试用例：CASE_01, CASE_02, CASE_05

import torch
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Helper functions (从主文件复制)
def create_tensor(device='cpu', dtype=torch.float32, requires_grad=True):
    """Create a test tensor with specified properties."""
    if dtype == torch.float32:
        data = torch.randn(3, 4, device=device, dtype=dtype)
    elif dtype == torch.float64:
        data = torch.randn(3, 4, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    if requires_grad:
        data.requires_grad_(True)
    return data

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize("device,dtype,requires_grad_input", [
    ("cpu", torch.float32, True),
    ("cpu", torch.float64, False),
])
def test_no_grad_basic(device, dtype, requires_grad_input):
    """TC-01: no_grad基础功能
    
    测试no_grad上下文管理器的基本功能：
    1. 在no_grad上下文内创建的张量requires_grad=False
    2. 退出上下文后梯度计算状态恢复
    3. 无异常抛出
    
    Weak asserts:
    - requires_grad_false: 上下文内张量requires_grad=False
    - state_restored: 退出后梯度状态恢复
    - no_exception: 无异常抛出
    """
    # 记录初始状态
    initial_grad_enabled = torch.is_grad_enabled()
    
    # 创建输入张量
    input_tensor = create_tensor(device=device, dtype=dtype, requires_grad=requires_grad_input)
    
    # 测试no_grad上下文
    with torch.no_grad():
        # 断言1: 在no_grad上下文内梯度计算被禁用
        assert not torch.is_grad_enabled(), "Gradient computation should be disabled in no_grad context"
        
        # 创建新张量
        output_tensor = input_tensor * 2
        
        # 断言2: 在no_grad上下文内创建的新张量requires_grad=False
        assert not output_tensor.requires_grad, (
            f"Tensor created in no_grad context should have requires_grad=False, "
            f"but got {output_tensor.requires_grad}"
        )
        
        # 即使输入张量requires_grad=True，输出张量也不应该有梯度
        if requires_grad_input:
            assert input_tensor.requires_grad, "Input tensor should have requires_grad=True"
            assert not output_tensor.requires_grad, "Output tensor should not require grad even if input does"
        
        # 执行一些计算操作
        result = output_tensor.sum()
        
        # 断言3: 可以正常执行计算操作
        assert result is not None, "Computation should succeed in no_grad context"
    
    # 断言4: 退出上下文后梯度计算状态恢复
    assert torch.is_grad_enabled() == initial_grad_enabled, (
        f"Gradient enabled state should be restored to initial value {initial_grad_enabled}, "
        f"but got {torch.is_grad_enabled()}"
    )
    
    # 断言5: 可以正常创建新的张量（状态已恢复）
    new_tensor = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype, requires_grad=True)
    assert new_tensor.requires_grad == True, "New tensor should be able to require grad after exiting no_grad"
    
    # 清理
    del input_tensor, output_tensor, result, new_tensor
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("device,dtype,nesting_order", [
    ("cpu", torch.float32, "no_grad_inside_enable"),
    ("cpu", torch.float32, "enable_inside_no_grad"),
])
def test_enable_grad_interaction(device, dtype, nesting_order):
    """TC-02: enable_grad与no_grad交互
    
    测试enable_grad和no_grad上下文管理器的交互：
    1. 嵌套上下文的状态传播
    2. 正确恢复状态
    3. 无异常抛出
    
    Weak asserts:
    - state_propagation: 嵌套上下文状态正确传播
    - correct_restoration: 状态正确恢复
    - no_exception: 无异常抛出
    """
    # 记录初始状态
    initial_grad_enabled = torch.is_grad_enabled()
    
    # 创建测试张量
    test_tensor = create_tensor(device=device, dtype=dtype, requires_grad=True)
    
    if nesting_order == "no_grad_inside_enable":
        # 测试场景：no_grad嵌套在enable_grad内部
        with torch.enable_grad():
            # 外层：enable_grad应该启用梯度计算
            assert torch.is_grad_enabled(), "enable_grad should enable gradient computation"
            
            # 在外层创建张量
            outer_tensor = test_tensor * 2
            assert outer_tensor.requires_grad, "Tensor created in enable_grad context should require grad"
            
            with torch.no_grad():
                # 内层：no_grad应该禁用梯度计算
                assert not torch.is_grad_enabled(), "no_grad should disable gradient computation inside enable_grad"
                
                # 在内层创建张量
                inner_tensor = test_tensor * 3
                assert not inner_tensor.requires_grad, "Tensor created in no_grad context should not require grad"
                
                # 可以访问外层张量
                combined = outer_tensor + inner_tensor
                assert not combined.requires_grad, "Operation with no_grad tensor should not require grad"
            
            # 退出内层后：应该恢复到enable_grad状态
            assert torch.is_grad_enabled(), "Should return to enable_grad state after exiting no_grad"
            
            # 在外层创建新张量
            new_outer_tensor = test_tensor * 4
            assert new_outer_tensor.requires_grad, "New tensor in enable_grad context should require grad"
        
        # 退出外层后：应该恢复到初始状态
        assert torch.is_grad_enabled() == initial_grad_enabled, "Should restore to initial state after exiting enable_grad"
    
    elif nesting_order == "enable_inside_no_grad":
        # 测试场景：enable_grad嵌套在no_grad内部
        with torch.no_grad():
            # 外层：no_grad应该禁用梯度计算
            assert not torch.is_grad_enabled(), "no_grad should disable gradient computation"
            
            # 在外层创建张量
            outer_tensor = test_tensor * 2
            assert not outer_tensor.requires_grad, "Tensor created in no_grad context should not require grad"
            
            with torch.enable_grad():
                # 内层：enable_grad应该启用梯度计算
                assert torch.is_grad_enabled(), "enable_grad should enable gradient computation inside no_grad"
                
                # 在内层创建张量
                inner_tensor = test_tensor * 3
                assert inner_tensor.requires_grad, "Tensor created in enable_grad context should require grad"
                
                # 可以访问外层张量
                combined = outer_tensor + inner_tensor
                # 注意：当与no_grad张量运算时，结果可能不require grad
                # 这是PyTorch的预期行为
            
            # 退出内层后：应该恢复到no_grad状态
            assert not torch.is_grad_enabled(), "Should return to no_grad state after exiting enable_grad"
            
            # 在外层创建新张量
            new_outer_tensor = test_tensor * 4
            assert not new_outer_tensor.requires_grad, "New tensor in no_grad context should not require grad"
        
        # 退出外层后：应该恢复到初始状态
        assert torch.is_grad_enabled() == initial_grad_enabled, "Should restore to initial state after exiting no_grad"
    
    else:
        raise ValueError(f"Unknown nesting_order: {nesting_order}")
    
    # 最终状态验证
    assert torch.is_grad_enabled() == initial_grad_enabled, (
        f"Final gradient enabled state should be {initial_grad_enabled}, "
        f"but got {torch.is_grad_enabled()}"
    )
    
    # 清理
    del test_tensor
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_05 START ====
# Placeholder for CASE_05: 装饰器用法验证
# TC-05: 装饰器用法验证
# Parameters: device=cpu, dtype=float32, decorator_type=no_grad/inference_mode
# Weak asserts: decorator_wraps, function_callable, grad_state_correct
# ==== BLOCK:CASE_05 END ====

if __name__ == "__main__":
    # Simple test runner for debugging
    import sys
    pytest.main([__file__] + sys.argv[1:])