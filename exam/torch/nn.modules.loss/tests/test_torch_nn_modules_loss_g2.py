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
    @pytest.mark.parametrize("reduction, input_shape, target_shape, dtype, value_range", [
        ("mean", (3, 2), (3, 2), torch.float32, [0, 1]),  # 基础测试用例
        ("sum", (2, 3, 4), (2, 3, 4), torch.float64, [0.1, 0.9]),  # 参数扩展：3D形状和不同dtype
    ])
    def test_bceloss_probability_input_validation(self, reduction, input_shape, target_shape, dtype, value_range):
        """测试BCELoss概率输入验证 (TC-06)"""
        # 创建输入张量（概率值，在[0,1]范围内）
        min_val, max_val = value_range
        input_tensor = torch.rand(input_shape, dtype=dtype) * (max_val - min_val) + min_val
        
        # 创建目标张量（二值标签，0或1）
        # 对于BCELoss，目标应该是0或1，但也可以是概率值
        target_tensor = torch.randint(0, 2, target_shape, dtype=dtype)
        
        # 创建损失函数实例
        loss_fn = nn.BCELoss(reduction=reduction)
        
        # 计算损失
        loss = loss_fn(input_tensor, target_tensor)
        
        # 手动计算参考值
        expected_loss = self.compute_bce_loss_manual(input_tensor, target_tensor, reduction)
        
        # weak断言验证
        # 1. 输出在有效范围内（BCE损失应该是非负的）
        if reduction != 'none':
            assert loss.item() >= 0, f"BCE loss should be non-negative, got {loss.item()}"
        else:
            assert torch.all(loss >= 0), f"BCE loss elements should be non-negative"
        
        # 2. 概率输入处理验证
        # 验证输入在有效范围内（BCELoss内部会进行clamping）
        assert torch.all(input_tensor >= 0), f"Input should be non-negative for BCE loss"
        assert torch.all(input_tensor <= 1), f"Input should be <= 1 for BCE loss"
        
        # 3. 有限值检查
        assert torch.isfinite(loss).all(), f"Loss contains non-finite values: {loss}"
        
        # 4. 输出形状匹配reduction模式
        if reduction == 'none':
            assert loss.shape == input_shape, f"Expected shape {input_shape}, got {loss.shape}"
        elif reduction == 'mean' or reduction == 'sum':
            assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
        
        # 5. 数据类型正确
        assert loss.dtype == dtype, f"Expected dtype {dtype}, got {loss.dtype}"
        
        # 6. 数值正确性验证（使用容差）
        # 注意：BCELoss内部使用log函数，可能会有数值误差，使用稍大的容差
        if reduction == 'none':
            assert torch.allclose(loss, expected_loss, rtol=1e-4, atol=1e-5), \
                f"Loss values don't match for reduction='none'"
        else:
            assert torch.allclose(loss, expected_loss, rtol=1e-4, atol=1e-5), \
                f"Loss value doesn't match for reduction='{reduction}': expected {expected_loss}, got {loss}"
        
        # 7. 验证边界值处理
        # 测试输入接近0或1的情况（BCELoss应该能处理）
        if min_val == 0 and max_val == 1:
            # 创建边界值测试
            boundary_input = torch.tensor([[0.0, 1.0], [0.5, 0.5]], dtype=dtype)
            boundary_target = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
            
            boundary_loss = loss_fn(boundary_input, boundary_target)
            assert torch.isfinite(boundary_loss).all(), \
                f"BCELoss should handle boundary values (0 and 1)"
        
        # 8. 验证不同reduction模式之间的关系
        if reduction == 'none':
            # 验证逐元素BCE计算
            eps = 1e-12
            input_clamped = torch.clamp(input_tensor, eps, 1 - eps)
            manual_bce = -(target_tensor * torch.log(input_clamped) + 
                          (1 - target_tensor) * torch.log(1 - input_clamped))
            assert torch.allclose(loss, manual_bce, rtol=1e-4, atol=1e-5)
        elif reduction == 'mean':
            # 验证mean是none的平均值
            none_loss_fn = nn.BCELoss(reduction='none')
            none_loss = none_loss_fn(input_tensor, target_tensor)
            mean_from_none = torch.mean(none_loss)
            assert torch.allclose(loss, mean_from_none, rtol=1e-4, atol=1e-5)
        elif reduction == 'sum':
            # 验证sum是none的总和
            none_loss_fn = nn.BCELoss(reduction='none')
            none_loss = none_loss_fn(input_tensor, target_tensor)
            sum_from_none = torch.sum(none_loss)
            assert torch.allclose(loss, sum_from_none, rtol=1e-4, atol=1e-5)
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
    @pytest.mark.parametrize("reduction, input_shape, target_shape, dtype", [
        ("mean", (4, 5), (4,), torch.float32),  # 基础测试用例
    ])
    def test_nllloss_negative_log_likelihood_validation(self, reduction, input_shape, target_shape, dtype):
        """测试NLLLoss负对数似然验证 (TC-07)"""
        # 创建输入张量（log probabilities）
        # NLLLoss期望输入是log probabilities，所以我们对随机值取log_softmax
        raw_input = torch.randn(input_shape, dtype=dtype)
        input_tensor = F.log_softmax(raw_input, dim=1)
        
        # 创建目标张量（类别索引）
        # 对于NLLLoss，目标应该是类别索引，范围在[0, num_classes-1]
        num_classes = input_shape[1]
        target_tensor = torch.randint(0, num_classes, target_shape, dtype=torch.long)
        
        # 创建损失函数实例
        loss_fn = nn.NLLLoss(reduction=reduction)
        
        # 计算损失
        loss = loss_fn(input_tensor, target_tensor)
        
        # 手动计算参考值
        expected_loss = self.compute_nll_loss_manual(input_tensor, target_tensor, reduction)
        
        # weak断言验证
        # 1. 输出是标量（对于mean/sum reduction）或向量（对于none reduction）
        if reduction == 'mean' or reduction == 'sum':
            assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
        elif reduction == 'none':
            # 对于none reduction，输出形状应该与batch size相同
            expected_none_shape = torch.Size([input_shape[0]])
            assert loss.shape == expected_none_shape, \
                f"Expected shape {expected_none_shape}, got {loss.shape}"
        
        # 2. 输入是log概率验证
        # 验证输入是log probabilities（每行的和应该接近0，因为log_softmax后exp和=1）
        # 实际上，log_softmax确保每行的log probabilities exp后和为1
        exp_input = torch.exp(input_tensor)
        row_sums = torch.sum(exp_input, dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5), \
            "Input should be log probabilities (exp should sum to 1 per row)"
        
        # 3. 类别索引目标验证
        # 验证目标索引在有效范围内
        assert torch.all(target_tensor >= 0), f"Target indices should be non-negative"
        assert torch.all(target_tensor < num_classes), \
            f"Target indices should be < {num_classes}, got max {torch.max(target_tensor)}"
        
        # 4. 有限输出检查
        assert torch.isfinite(loss).all(), f"Loss contains non-finite values: {loss}"
        
        # 5. 数值正确性验证（使用容差）
        assert torch.allclose(loss, expected_loss, rtol=1e-5, atol=1e-5), \
            f"Loss value doesn't match: expected {expected_loss}, got {loss}"
        
        # 6. 验证损失值非负（NLL损失总是非负，因为概率<=1，所以-log(prob)>=0）
        if reduction != 'none':
            assert loss.item() >= 0, f"NLL loss should be non-negative, got {loss.item()}"
        else:
            assert torch.all(loss >= 0), f"NLL loss elements should be non-negative"
        
        # 7. 验证输入形状兼容性
        # 输入应该是(batch_size, num_classes, ...)的形式
        assert input_tensor.dim() >= 2, f"Input should have at least 2 dimensions, got {input_tensor.dim()}"
        assert input_tensor.size(1) == num_classes, \
            f"Input second dimension should be num_classes={num_classes}, got {input_tensor.size(1)}"
        
        # 8. 验证目标形状兼容性
        # 目标应该是(batch_size, ...)的形式，没有类别维度
        assert target_tensor.dim() == input_tensor.dim() - 1, \
            f"Target should have one less dimension than input: input dim={input_tensor.dim()}, target dim={target_tensor.dim()}"
        assert target_tensor.size(0) == input_tensor.size(0), \
            f"Batch size mismatch: input batch={input_tensor.size(0)}, target batch={target_tensor.size(0)}"
        
        # 9. 验证不同reduction模式之间的关系
        if reduction == 'none':
            # 验证逐元素NLL计算
            manual_nll = -input_tensor[range(len(target_tensor)), target_tensor]
            assert torch.allclose(loss, manual_nll, rtol=1e-5, atol=1e-5)
        elif reduction == 'mean':
            # 验证mean是none的平均值
            none_loss_fn = nn.NLLLoss(reduction='none')
            none_loss = none_loss_fn(input_tensor, target_tensor)
            mean_from_none = torch.mean(none_loss)
            assert torch.allclose(loss, mean_from_none, rtol=1e-5, atol=1e-5)
        elif reduction == 'sum':
            # 验证sum是none的总和
            none_loss_fn = nn.NLLLoss(reduction='none')
            none_loss = none_loss_fn(input_tensor, target_tensor)
            sum_from_none = torch.sum(none_loss)
            assert torch.allclose(loss, sum_from_none, rtol=1e-5, atol=1e-5)
        
        # 10. 验证与CrossEntropyLoss的关系
        # NLLLoss + log_softmax = CrossEntropyLoss
        if reduction == 'mean':
            ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
            ce_loss = ce_loss_fn(raw_input, target_tensor)
            assert torch.allclose(loss, ce_loss, rtol=1e-5, atol=1e-5), \
                f"NLLLoss should equal CrossEntropyLoss for same input: NLL={loss}, CE={ce_loss}"
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
    def test_invalid_reduction_parameter_g2(self):
        """测试G2组损失函数的无效reduction参数"""
        # 根据PyTorch的_Reduction.get_enum函数，无效的reduction参数会抛出ValueError
        with pytest.raises(ValueError, match=".*is not a valid value for reduction.*"):
            nn.BCELoss(reduction='invalid')
        
        with pytest.raises(ValueError, match=".*is not a valid value for reduction.*"):
            nn.NLLLoss(reduction='wrong')
        
        with pytest.raises(ValueError, match=".*is not a valid value for reduction.*"):
            nn.KLDivLoss(reduction='bad')
        
        # 测试一些边界情况
        # 空字符串
        with pytest.raises(ValueError):
            nn.BCELoss(reduction='')
        
        # None值（应该使用默认值'mean'，而不是引发错误）
        # 注意：reduction参数有默认值'mean'，所以传递None应该使用默认值
        loss_fn = nn.BCELoss(reduction=None)
        assert loss_fn.reduction == 'mean', f"Default reduction should be 'mean', got {loss_fn.reduction}"
    
    def test_shape_mismatch_errors_g2(self):
        """测试G2组损失函数的形状不匹配错误"""
        # 根据PyTorch的binary_cross_entropy函数，形状不匹配会抛出ValueError
        # BCELoss形状不匹配
        loss_fn = nn.BCELoss()
        input_tensor = torch.rand(2, 3, 4)
        target_tensor = torch.rand(2, 3, 5)  # 形状不匹配
        
        # 根据binary_cross_entropy源码，形状不匹配会抛出ValueError
        with pytest.raises(ValueError, match=".*target size.*different to the input size.*"):
            loss_fn(input_tensor, target_tensor)
        
        # NLLLoss形状不匹配
        loss_fn = nn.NLLLoss()
        input_tensor = torch.randn(2, 5, 3, 4)  # log probabilities
        target_tensor = torch.randint(0, 5, (2, 3, 5))  # 形状不匹配
        
        # NLLLoss也会检查形状兼容性
        with pytest.raises(ValueError):
            loss_fn(input_tensor, target_tensor)
        
        # KLDivLoss形状不匹配
        loss_fn = nn.KLDivLoss()
        input_tensor = F.log_softmax(torch.randn(3, 4), dim=1)
        target_tensor = F.softmax(torch.randn(3, 5), dim=1)  # 形状不匹配
        
        with pytest.raises(RuntimeError):
            loss_fn(input_tensor, target_tensor)
    
    def test_bceloss_invalid_probability_range(self):
        """测试BCELoss输入概率范围验证"""
        loss_fn = nn.BCELoss()
        
        # 输入超出[0,1]范围 - BCELoss应该能处理（内部会clamp到有效范围）
        # 根据PyTorch实现，BCELoss内部使用binary_cross_entropy，它会处理超出范围的输入
        input_tensor = torch.tensor([[1.5, -0.5], [0.3, 0.7]])
        target_tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        # BCELoss应该能处理超出范围的输入（内部会clamp到[eps, 1-eps]）
        loss = loss_fn(input_tensor, target_tensor)
        assert torch.isfinite(loss), f"BCELoss should handle out-of-range inputs, got {loss}"
        
        # 验证损失值是非负的
        assert loss.item() >= 0, f"BCE loss should be non-negative, got {loss.item()}"
        
        # 测试极端值
        extreme_input = torch.tensor([[100.0, -100.0], [0.0, 1.0]])
        extreme_loss = loss_fn(extreme_input, target_tensor)
        assert torch.isfinite(extreme_loss), f"BCELoss should handle extreme values"
        
        # 验证与手动计算的一致性（手动计算会进行clamp）
        eps = 1e-12
        input_clamped = torch.clamp(input_tensor, eps, 1 - eps)
        manual_bce = -(target_tensor * torch.log(input_clamped) + 
                      (1 - target_tensor) * torch.log(1 - input_clamped))
        manual_loss = torch.mean(manual_bce)
        
        # 由于clamping，可能会有一些数值差异，但应该在合理范围内
        assert torch.allclose(loss, manual_loss, rtol=1e-4, atol=1e-5), \
            f"BCELoss should produce similar results to clamped manual calculation: {loss} vs {manual_loss}"
    
    def test_nllloss_invalid_target_indices(self):
        """测试NLLLoss无效目标索引"""
        loss_fn = nn.NLLLoss()
        input_tensor = F.log_softmax(torch.randn(3, 5), dim=1)  # 3个样本，5个类别
        
        # 目标索引超出范围
        target_tensor = torch.tensor([5, 1, 2], dtype=torch.long)  # 5超出范围[0,4]
        
        # NLLLoss应该对无效索引抛出错误
        # 注意：具体错误类型可能因PyTorch版本而异
        with pytest.raises((RuntimeError, IndexError)):
            loss_fn(input_tensor, target_tensor)
        
        # 测试负索引
        target_tensor = torch.tensor([-1, 1, 2], dtype=torch.long)
        with pytest.raises((RuntimeError, IndexError)):
            loss_fn(input_tensor, target_tensor)
    
    def test_kldivloss_log_target_parameter(self):
        """测试KLDivLoss的log_target参数"""
        # 测试log_target=True的情况
        loss_fn = nn.KLDivLoss(reduction='mean', log_target=True)
        
        # 创建log probabilities输入
        input_tensor = F.log_softmax(torch.randn(3, 4), dim=1)
        target_tensor = F.log_softmax(torch.randn(3, 4), dim=1)
        
        # 应该能正常计算
        loss = loss_fn(input_tensor, target_tensor)
        assert torch.isfinite(loss), f"KLDivLoss with log_target=True should work, got {loss}"
        
        # 验证损失值是非负的（KL散度总是非负）
        assert loss.item() >= 0, f"KL divergence should be non-negative, got {loss.item()}"
        
        # 测试log_target=False的情况
        loss_fn_no_log = nn.KLDivLoss(reduction='mean', log_target=False)
        target_prob = F.softmax(torch.randn(3, 4), dim=1)
        
        loss_no_log = loss_fn_no_log(input_tensor, target_prob)
        assert torch.isfinite(loss_no_log), f"KLDivLoss with log_target=False should work"
        
        # 验证两种模式在数学上应该等价（当target是log probabilities时）
        # 注意：由于数值精度，可能会有微小差异
        target_exp = torch.exp(target_tensor)
        loss_manual = loss_fn_no_log(input_tensor, target_exp)
        assert torch.allclose(loss, loss_manual, rtol=1e-5, atol=1e-5), \
            f"log_target=True should be equivalent to exp(target) with log_target=False"
    
    def test_loss_functions_are_callable_g2(self):
        """测试G2组损失函数是可调用的"""
        # 测试所有G2组的损失函数
        loss_classes = [nn.BCELoss, nn.NLLLoss, nn.KLDivLoss]
        
        for loss_class in loss_classes:
            # 创建实例
            loss_fn = loss_class()
            
            # 创建测试数据
            if loss_class == nn.NLLLoss:
                input_tensor = F.log_softmax(torch.randn(2, 5), dim=1)
                target_tensor = torch.randint(0, 5, (2,), dtype=torch.long)
            elif loss_class == nn.KLDivLoss:
                input_tensor = F.log_softmax(torch.randn(2, 5), dim=1)
                target_tensor = F.softmax(torch.randn(2, 5), dim=1)
            else:  # BCELoss
                input_tensor = torch.rand(2, 3)
                target_tensor = torch.randint(0, 2, (2, 3), dtype=torch.float32)
            
            # 验证可调用
            assert callable(loss_fn), f"{loss_class.__name__} should be callable"
            
            # 验证调用返回张量
            result = loss_fn(input_tensor, target_tensor)
            assert isinstance(result, torch.Tensor), \
                f"{loss_class.__name__} should return a Tensor, got {type(result)}"
            
            # 验证返回的张量有正确的属性
            assert result.dtype in [torch.float32, torch.float64], \
                f"Loss should return float tensor, got {result.dtype}"
            assert result.requires_grad == False, \
                f"Loss output should not require grad by default"
    
    def test_batchmean_reduction_kldivloss(self):
        """测试KLDivLoss的batchmean reduction模式"""
        # KLDivLoss支持特殊的'batchmean' reduction
        loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        # 创建输入和目标（log probabilities和probabilities）
        input_tensor = F.log_softmax(torch.randn(3, 4), dim=1)
        target_tensor = F.softmax(torch.randn(3, 4), dim=1)
        
        # 计算损失
        loss = loss_fn(input_tensor, target_tensor)
        
        # 验证输出是标量
        assert loss.shape == torch.Size([]), f"Expected scalar for batchmean reduction, got shape {loss.shape}"
        
        # 验证有限值
        assert torch.isfinite(loss), f"KLDivLoss batchmean should produce finite value, got {loss}"
        
        # 验证batchmean是sum除以batch size
        sum_loss_fn = nn.KLDivLoss(reduction='sum')
        sum_loss = sum_loss_fn(input_tensor, target_tensor)
        expected_batchmean = sum_loss / 3  # batch size = 3
        
        assert torch.allclose(loss, expected_batchmean, rtol=1e-5, atol=1e-5), \
            f"batchmean should be sum divided by batch size: expected {expected_batchmean}, got {loss}"
        
        # 验证batchmean与其他reduction模式的关系
        none_loss_fn = nn.KLDivLoss(reduction='none')
        none_loss = none_loss_fn(input_tensor, target_tensor)
        
        # batchmean应该是所有元素的和除以batch size
        manual_batchmean = torch.sum(none_loss) / 3
        assert torch.allclose(loss, manual_batchmean, rtol=1e-5, atol=1e-5), \
            f"batchmean should equal sum(none) / batch_size"
        
        # 验证batchmean是mean乘以元素数量除以batch size
        mean_loss_fn = nn.KLDivLoss(reduction='mean')
        mean_loss = mean_loss_fn(input_tensor, target_tensor)
        total_elements = 3 * 4  # batch_size * num_elements_per_sample
        expected_from_mean = mean_loss * total_elements / 3
        assert torch.allclose(loss, expected_from_mean, rtol=1e-5, atol=1e-5), \
            f"batchmean should equal mean * total_elements / batch_size"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====