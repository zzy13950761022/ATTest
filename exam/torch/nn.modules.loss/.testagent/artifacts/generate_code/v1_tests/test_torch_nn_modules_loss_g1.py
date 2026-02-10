"""
测试 torch.nn.modules.loss 模块中的核心损失函数族（G1组）
包含：L1Loss, MSELoss, CrossEntropyLoss
"""

import math
import warnings
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# ==== BLOCK:HEADER START ====
# 测试类定义和通用辅助函数
class TestLossFunctionsG1:
    """测试核心损失函数族（G1组）"""
    
    @staticmethod
    def compute_l1_loss_manual(input_tensor, target_tensor, reduction='mean'):
        """手动计算L1损失作为参考"""
        diff = torch.abs(input_tensor - target_tensor)
        if reduction == 'none':
            return diff
        elif reduction == 'mean':
            return torch.mean(diff)
        elif reduction == 'sum':
            return torch.sum(diff)
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
    
    @staticmethod
    def compute_mse_loss_manual(input_tensor, target_tensor, reduction='mean'):
        """手动计算MSE损失作为参考"""
        diff = input_tensor - target_tensor
        squared = diff * diff
        if reduction == 'none':
            return squared
        elif reduction == 'mean':
            return torch.mean(squared)
        elif reduction == 'sum':
            return torch.sum(squared)
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
    
    @staticmethod
    def compute_cross_entropy_manual(input_tensor, target_tensor, reduction='mean'):
        """手动计算交叉熵损失作为参考"""
        # 对输入进行softmax
        log_softmax = F.log_softmax(input_tensor, dim=1)
        # 获取目标类别的log概率
        nll_loss = -log_softmax[range(len(target_tensor)), target_tensor]
        
        if reduction == 'none':
            return nll_loss
        elif reduction == 'mean':
            return torch.mean(nll_loss)
        elif reduction == 'sum':
            return torch.sum(nll_loss)
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# L1Loss基础功能验证
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# MSELoss三种reduction模式
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
# CrossEntropyLoss形状兼容性
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# L1Loss弃用参数向后兼容（占位）
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# 极端形状与数值边界（占位）
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:FOOTER START ====
# 清理和额外测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====