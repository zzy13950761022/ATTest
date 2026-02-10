#!/usr/bin/env python3
"""验证修复的测试"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_l1loss_shape_mismatch():
    """测试L1Loss形状不匹配错误"""
    loss_fn = nn.L1Loss()
    input_tensor = torch.randn(2, 3, 4)
    target_tensor = torch.randn(2, 3, 5)  # 形状不匹配
    
    try:
        loss_fn(input_tensor, target_tensor)
        print("ERROR: L1Loss should raise RuntimeError for shape mismatch")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        print(f"L1Loss shape mismatch error: {error_msg}")
        # 检查错误消息是否包含预期的关键词
        if "must match" in error_msg and "size" in error_msg and "tensor" in error_msg:
            print("✓ L1Loss shape mismatch error message is correct")
            return True
        else:
            print(f"ERROR: L1Loss error message doesn't match pattern: {error_msg}")
            return False

def test_bceloss_invalid_range():
    """测试BCELoss输入超出范围"""
    loss_fn = nn.BCELoss()
    input_tensor = torch.tensor([[1.5, -0.5], [0.3, 0.7]])
    target_tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    try:
        loss_fn(input_tensor, target_tensor)
        print("ERROR: BCELoss should raise RuntimeError for out-of-range input")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        print(f"BCELoss out-of-range error: {error_msg}")
        # 检查错误消息是否包含预期的关键词
        if "all elements of input should be between 0 and 1" in error_msg:
            print("✓ BCELoss out-of-range error message is correct")
            return True
        else:
            print(f"ERROR: BCELoss error message doesn't match pattern: {error_msg}")
            return False

def test_nllloss_shape_mismatch():
    """测试NLLLoss形状不匹配错误"""
    loss_fn = nn.NLLLoss()
    input_tensor = torch.randn(2, 5, 3, 4)  # log probabilities
    target_tensor = torch.randint(0, 5, (2, 3, 5))  # 形状不匹配
    
    try:
        loss_fn(input_tensor, target_tensor)
        print("ERROR: NLLLoss should raise RuntimeError for shape mismatch")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        print(f"NLLLoss shape mismatch error: {error_msg}")
        # 检查错误消息是否包含预期的关键词
        if "size mismatch" in error_msg:
            print("✓ NLLLoss shape mismatch error message is correct")
            return True
        else:
            print(f"ERROR: NLLLoss error message doesn't match pattern: {error_msg}")
            return False

def main():
    print("=" * 60)
    print("验证修复的测试")
    print("=" * 60)
    
    results = []
    
    print("\n1. 测试L1Loss形状不匹配错误:")
    results.append(test_l1loss_shape_mismatch())
    
    print("\n2. 测试BCELoss输入超出范围:")
    results.append(test_bceloss_invalid_range())
    
    print("\n3. 测试NLLLoss形状不匹配错误:")
    results.append(test_nllloss_shape_mismatch())
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    for i, result in enumerate(results):
        test_name = ["L1Loss形状不匹配", "BCELoss输入超出范围", "NLLLoss形状不匹配"][i]
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{i+1}. {test_name}: {status}")
    
    all_passed = all(results)
    print(f"\n总体结果: {'所有测试通过' if all_passed else '有测试失败'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())