"""
测试 torch.nn.modules.loss 模块中的加权与特殊损失函数族（G2组）
包含：BCELoss, NLLLoss, KLDivLoss
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
class TestLossFunctionsG2:
    """测试加权与特殊损失函数族（G2组）"""
    
    @staticmethod
    def compute_bce_loss_manual(input_tensor, target_tensor, reduction='mean'):
        """手动计算二值交叉熵损失作为参考"""
        # 确保输入在[0,1]范围内（BCELoss要求）
        eps = 1e-12
        input_clamped = torch.clamp(input_tensor, eps, 1 - eps)
        
        # 计算二值交叉熵
        bce = -(target_tensor * torch.log(input_clamped) + 
                (1 - target_tensor) * torch.log(1 - input_clamped))
        
        if reduction == 'none':
            return bce
        elif reduction == 'mean':
            return torch.mean(bce)
        elif reduction == 'sum':
            return torch.sum(bce)
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
    
    @staticmethod
    def compute_nll_loss_manual(input_tensor, target_tensor, reduction='mean'):
        """手动计算负对数似然损失作为参考"""
        # 输入应该是log probabilities
        # 目标应该是类别索引
        nll = -input_tensor[range(len(target_tensor)), target_tensor]
        
        if reduction == 'none':
            return nll
        elif reduction == 'mean':
            return torch.mean(nll)
        elif reduction == 'sum':
            return torch.sum(nll)
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
    
    @staticmethod
    def compute_kl_divergence_manual(input_tensor, target_tensor, reduction='mean', log_target=False):
        """手动计算KL散度作为参考"""
        if log_target:
            # 如果目标是log probabilities
            kl = target_tensor.exp() * (target_tensor - input_tensor)
        else:
            # 如果目标是probabilities
            eps = 1e-12
            input_clamped = torch.clamp(input_tensor, eps, 1)
            target_clamped = torch.clamp(target_tensor, eps, 1)
            kl = target_clamped * (torch.log(target_clamped) - torch.log(input_clamped))
        
        if reduction == 'none':
            return kl
        elif reduction == 'mean':
            return torch.mean(kl)
        elif reduction == 'sum':
            return torch.sum(kl)
        elif reduction == 'batchmean':
            # batchmean是sum除以batch size
            return torch.sum(kl) / kl.size(0)
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_06 START ====
# BCELoss概率输入验证（占位）
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# NLLLoss负对数似然验证（占位）
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# KLDivLoss特殊reduction处理（占位）
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:CASE_09 START ====
# 加权损失类权重参数（占位）
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:CASE_10 START ====
# 设备兼容性验证（占位）
# ==== BLOCK:CASE_10 END ====

# ==== BLOCK:FOOTER START ====
# 通用错误测试和辅助函数（占位）
# ==== BLOCK:FOOTER END ====